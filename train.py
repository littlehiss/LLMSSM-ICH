import os
import sys
import json
import shutil
import numpy as np
import random
import time
import math
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from MedMamba import VSSM as medmamba # import model
from dataset import dataset2D, screch_excel
from transformers import CLIPModel
from build_model import build_model
from segment_anything import sam_model_registry, SamPredictor
import models.resnet as models
from tools import AverageMeter, adjust_learning_rate, accuracy, save_checkpoint

seed_value = 3407

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True

current_time = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

def kd_loss_function(output, target_output,args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def step_decay(epoch, learning_rate, drop, epochs_drop):
    """
    learning rate step decay
    :param epoch: current training epoch
    :param learning_rate: initial learning rate
    :return: learning rate after step decay
    """
    initial_lrate = learning_rate
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

def get_train_test_split(good_patients, bad_patients, good_patients_dict, bad_patients_dict, kf):
    splits = []

    good_indices = list(range(len(good_patients)))
    good_splits = list(kf.split(good_indices))

    bad_indices = list(range(len(bad_patients)))
    bad_splits = list(kf.split(bad_indices))

    for i in range(5):
        train_index_good, test_index_good = good_splits[i]
        train_index_bad, test_index_bad = bad_splits[i]

        train_good = [good_patients_dict[good_patients[i]] for i in train_index_good]
        test_good = [good_patients_dict[good_patients[i]] for i in test_index_good]

        train_bad = [bad_patients_dict[bad_patients[i]] for i in train_index_bad]
        test_bad = [bad_patients_dict[bad_patients[i]] for i in test_index_bad]

        train = train_good + train_bad
        test = test_good + test_bad

        splits.append((train, test))

    return splits

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    text_file_path = './ICH_data.xlsx'
    good_patients_dict = screch_excel(text_file_path, os.path.join(args.train_path, 'good'))
    bad_patients_dict = screch_excel(text_file_path, os.path.join(args.train_path, 'bad'))

    good_patients = list(good_patients_dict.keys())
    bad_patients = list(bad_patients_dict.keys())

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = get_train_test_split(good_patients, bad_patients, good_patients_dict, bad_patients_dict, kf)

    for fold, (train_set, validate_set) in enumerate(splits):
        train_data = dataset2D(train_set)
        validate_data = dataset2D(validate_set)
        train_num = len(train_data)
        val_num = len(validate_data)

        print(f"==========Fold {fold + 1}==========")
        print(f"Train patient number: {train_num} items")
        print(f"Validation patient number: {val_num} items")
        os.makedirs("./logs/" + current_time, exist_ok=True)
        checkpoint_path = os.path.join('./logs/' + current_time, 'fold_' + str(fold))
        os.makedirs(checkpoint_path, exist_ok=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, drop_last=True)
        validate_loader = torch.utils.data.DataLoader(dataset=validate_data, batch_size=args.batch_size)

        print("Prepare completed! Launch training!\U0001F680")
        sam_model = sam_model_registry[args.model_type](args)
        model = medmamba(num_classes=args.num_classes)

        if args.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model).to(device)

        train_steps = len(train_loader)
        for epoch in range(args.epochs):
            LEARNING_RATE = step_decay(epoch, args.lr, args.drop, args.epochs_drop)
            print(f'Learning Rate: {LEARNING_RATE}')
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            # train
            model.train()

            running_loss = 0.0
            train_acc = 0.0
            best_acc = 0.0

            train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
            for step, data in enumerate(train_bar):
                imgs, texts, label, patient_names = data
                imgs = imgs.to(device)
                texts = texts.to(device)
                label = label.to(device)
                outputs = model.module(imgs, texts)

                loss = criterion(outputs, label)
                predict_y = torch.max(outputs, dim=1)[1]
                train_acc += torch.eq(predict_y, label).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, running_loss)
            train_accurate = train_acc / train_num
            print("[epoch {}] train_loss: {:.4} train_accuracy: {:.5}".format(epoch + 1, running_loss / train_steps, train_accurate))

            # validate
            model.eval()

            acc = 0.0  # accumulate accurate number / epoch
            val_loss = 0.0
            TP = 0.0
            FP = 0.0
            TN = 0.0
            FN = 0.0
            name_list = []
            auc_gt = []
            auc_pred = []
            with torch.no_grad():
                val_bar = tqdm(validate_loader, leave=True, file=sys.stdout)
                for val_data in val_bar:
                    imgs, texts, label, patient_names = val_data
                    imgs = imgs.to(device)
                    texts = texts.to(device)
                    label = label.to(device)
                    name_list.append(str(patient_names))
                    outputs = model.module(imgs, texts)

                    loss = criterion(outputs, label)
                    val_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    auc_gt.extend(label.cpu().numpy())
                    auc_pred.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    acc += torch.eq(predict_y, label).sum().item()

                    # TP, FP, TN, FN
                    for j in range(len(label)):
                        if label[j] == 1 and predict_y[j] == 1:
                            TP += 1
                        elif label[j] == 0 and predict_y[j] == 1:
                            FP += 1
                        elif label[j] == 0 and predict_y[j] == 0:
                            TN += 1
                        elif label[j] == 1 and predict_y[j] == 0:
                            FN += 1

            val_accurate = (TP + TN) / (TP + FP + TN + FN)  # accuracy of each class
            val_loss = val_loss / val_num
            val_auc = roc_auc_score(auc_gt, auc_pred)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print("epoch {}: [Val: loss {:.5}, val acc {:.5}, val_AUC {:.5}, val_F1: {:.5}, val_precision: {:.5}, val_recall: {:.5}]".format(epoch + 1, float(val_loss), float(val_accurate), float(val_auc), float(f1_score), float(precision), float(recall)))
            with open(os.path.join('./logs/' + current_time, 'fold_' + str(fold) + '_test_loss_and_acc.txt'), 'a') as f:
                if epoch == 0:
                    f.write("-------\n" + "Patients:\n" + "\n".join(name_list) + "\n")
                f.write("epoch {}: [Val: loss {:.5}, val acc {:.5}, val_AUC {:.5}, val_F1: {:.5}, val_precision: {:.5}, val_recall: {:.5}]\n".format(epoch + 1, float(val_loss), float(val_accurate), float(val_auc), float(f1_score), float(precision), float(recall)))
            if (val_accurate > best_acc) and (val_accurate >= 80):
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'fold_' + str(fold) + '_epoch_' + str(epoch) + '.pth'))

    print('Finished Training!\U0001F40D')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with specific parameters.')
    parser.add_argument("--use_multi_gpu", default=True, help="Device choice", type=bool)
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for the model.')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs for training.')
    parser.add_argument('--drop', type=float, default=0.5, help='Learning rate drop.')
    parser.add_argument('--epochs_drop', type=float, default=30.0, help='Learning rate epoch drop.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=int, default=0.0001, help='Learning rate for training.')
    parser.add_argument('--model_name', type=str, default='SAM-GPT', help='Name of the model for saving.')
    parser.add_argument('--train_path', type=str, default='dataset/Sag-slicers', help='Dataset path.')
    parser.add_argument('--val_path', type=str, default='dataset/Sag-slicers', help='Dataset path.')
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")

    args = parser.parse_args()

    main(args)
