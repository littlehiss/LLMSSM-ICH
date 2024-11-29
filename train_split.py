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

seed_value = 3407   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

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

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

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

# def get_train_test_split(good_patients_dict, bad_patients_dict):
#     val_name = ['pat065', 'pat039', 'pat157', 'pat234', 'pat086', 'pat139', 'pat482', 'pat121', 'pat168', 'pat099',
#                 'pat328', 'pat082', 'pat119', 'pat165', 'pat452', 'pat385', 'pat403', 'pat392', 'pat308', 'pat451',
#                 'pat169', 'pat256', 'pat226', 'pat122', 'pat433', 'pat483', 'pat105', 'pat273', 'pat453', 'pat019',
#                 'pat100', 'pat447', 'pat439', 'pat219', 'pat477', 'pat258', 'pat240', 'pat313', 'pat340', 'pat280',
#                 'pat206', 'pat283', 'pat174', 'pat398', 'pat186', 'pat225', 'pat462', 'pat441', 'pat319', 'pat223',
#                 'pat014', 'pat201', 'pat030', 'pat410', 'pat217', 'pat362', 'pat033', 'pat090', 'pat351', 'pat017',
#                 'pat159', 'pat127', 'pat128', 'pat210', 'pat445', 'pat124', 'pat229', 'pat181', 'pat141', 'pat070',
#                 'pat272', 'pat146', 'pat115', 'pat299', 'pat051', 'pat335', 'pat138', 'pat028', 'pat361', 'pat209',
#                 'pat401']
#
#     # 获取患者ID列表
#     good_patients = list(good_patients_dict.keys())
#     bad_patients = list(bad_patients_dict.keys())
#
#     dataset_name = good_patients + bad_patients
#     dataset_dict = good_patients_dict.copy()
#     dataset_dict.update(bad_patients_dict)
#
#     dataset_train = []
#     for name in dataset_name:
#         if name not in val_name:
#             dataset_train.append(dataset_dict[name])
#     dataset_val = [dataset_dict[name] for name in val_name]
#
#     return dataset_train, dataset_val

def get_train_test_split(good_patients_dict, bad_patients_dict):
    # 设置随机种子以保证结果的可重复性
    random.seed(42)

    # 分别对good_patients和bad_patients进行8:2的划分
    def split_train_val(patient_dict, ratio=0.8):
        patient_keys = list(patient_dict.keys())
        random.shuffle(patient_keys)
        split_index = int(len(patient_keys) * ratio)
        train_keys = patient_keys[:split_index]
        val_keys = patient_keys[split_index:]
        train_data = [patient_dict[key] for key in train_keys]
        val_data = [patient_dict[key] for key in val_keys]
        return train_data, val_data

    # 分别对好坏患者进行划分
    good_train, good_val = split_train_val(good_patients_dict)
    bad_train, bad_val = split_train_val(bad_patients_dict)

    # 合并好患者和坏患者的训练集和验证集
    dataset_train = good_train + bad_train
    dataset_val = good_val + bad_val

    return dataset_train, dataset_val



def main(args):
    Fold_Num = 5
    for fold in range(Fold_Num):
        print("fold:", fold)

        seed = fold  # fold
        # seed = seed_lst[fold]  # fold
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # 构建患者信息字典
        text_file_path = './ICH_data.xlsx'  # 患者信息路径
        # patients_info = screch_excel(text_file_path, args.train_path)  # 获取患者的表格数据字典
        good_patients_dict = screch_excel(text_file_path, os.path.join(args.train_path, 'good'))
        bad_patients_dict = screch_excel(text_file_path, os.path.join(args.train_path, 'bad'))

        train_set, validate_set = get_train_test_split(good_patients_dict, bad_patients_dict)

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

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        validate_loader = torch.utils.data.DataLoader(dataset=validate_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

        print("Prepare completed! Launch training!\U0001F680")
        # sam_model = sam_model_registry[args.model_type](args)
        model = medmamba(num_classes=args.num_classes)
        # model = build_model().to(device)
        # clip_model = CLIPModel.from_pretrained("./pubmed-clip-vit-base-patch32").to(device)
        if args.use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model).to(device)
            # sam_model = nn.DataParallel(sam_model).to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer = optim.Adam(sam_model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay)

        best_acc = 0.0
        train_steps = len(train_loader)
        for epoch in range(args.epochs):
            start_time = time.time()
            LEARNING_RATE = step_decay(epoch, args.lr, args.drop, args.epochs_drop)
            print(f'Learning Rate: {LEARNING_RATE}')
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            # train
            model.train()
            # sam_model.train()

            # adjust_learning_rate(args, optimizer, epoch)  # 动态调整学习率

            running_loss = 0.0
            train_acc = 0.0

            train_auc_gt = []
            train_auc_pred = []

            train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
            for step, data in enumerate(train_bar):
                imgs, texts, label, patient_names = data
                imgs = imgs.to(device)
                texts = texts.to(device)
                label = label.to(device)

                # image_embeddings = sam_model.module.image_encoder(imgs)  # 使用SAM的编码器，并行时要通过module访问
                # outputs = sam_model.module.seg_fusion(image_embeddings, texts)  # 融合模块

                # outputs = model.module(imgs, last_hidden_state)
                outputs = model.module(imgs, texts)  # 原来mamba模型 GPT
                # outputs = model.module(imgs)
                # image_embeddings = image_embeddings.to(device)

                loss = criterion(outputs, label)
                predict_y = torch.max(outputs, dim=1)[1]
                train_acc += torch.eq(predict_y, label).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                # train_auc_gt.extend(label.cpu().numpy())
                # train_auc_pred.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                running_loss += loss.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, running_loss)
            train_accurate = train_acc / train_num
            # train_AUC = roc_auc_score(train_auc_gt, train_auc_pred)
            print("[epoch {}] train_loss: {:.4} train_accuracy: {:.5}".format(epoch + 1, running_loss / train_steps, train_accurate))


            # validate

            model.eval()
            # sam_model.eval()

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

                    # image_embeddings = sam_model.module.image_encoder(imgs)  # 使用SAM的编码器
                    # outputs = sam_model.module.seg_fusion(image_embeddings, texts)  # 融合模块
                    # outputs = model.module(val_imgs, val_last_hidden_state)
                    outputs = model.module(imgs, texts)

                    loss = criterion(outputs, label)
                    val_loss += loss.item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    auc_gt.extend(label.cpu().numpy())
                    auc_pred.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    acc += torch.eq(predict_y, label).sum().item()

                    # 更新 TP, FP, TN, FN
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
            print("epoch {}: [val loss {:.5}, val acc {:.5}, val_AUC {:.5}, val_F1: {:.5}, val_precision: {:.5}, val_recall: {:.5}]".format(epoch + 1, float(val_loss), float(val_accurate), float(val_auc), float(f1_score), float(precision), float(recall)))

            with open(os.path.join('./logs/' + current_time, 'fold_' + str(fold) + '_test_loss_and_acc.txt'), 'a') as f:
                if epoch == 0:
                    f.write("-------\n" + "Patients:\n" + "\n".join(name_list) + "\n")
                f.write("epoch {}: [val loss {:.5}, val acc {:.5}, val_AUC {:.5}, val_F1: {:.5}, val_precision: {:.5}, val_recall: {:.5}]\n".format(epoch + 1, float(val_loss), float(val_accurate), float(val_auc), float(f1_score), float(precision), float(recall)))
            # if val_accurate > best_acc:
            #     best_acc = val_accurate
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'fold_' + str(fold) + '_epoch_' + str(epoch + 1) + '.pth'))

    print('Finished Training!\U0001F40D')


if __name__ == '__main__':
    # 设置 argparse 解析器
    parser = argparse.ArgumentParser(description='Train a model with specific parameters.')
    parser.add_argument("--use_multi_gpu", default=True, help="Device choice", type=bool)
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for the model.')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for training.')
    parser.add_argument('--drop', type=float, default=0.1, help='Learning rate drop.')
    parser.add_argument('--epochs_drop', type=float, default=5.0, help='Learning rate epoch drop.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    # parser.add_argument('--lr', type=int, default=0.0001, help='Learning rate for training.')
    parser.add_argument('--lr', type=int, default=0.0001, help='Learning rate for training.')  # 0.0001
    parser.add_argument('--model_name', type=str, default='SAM-GPT', help='Name of the model for saving.')
    parser.add_argument('--train_path', type=str, default='dataset/Sag-slicers', help='Dataset path.')
    parser.add_argument('--val_path', type=str, default='dataset/Sag-slicers', help='Dataset path.')
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")

    # 解析命令行参数
    args = parser.parse_args()

    main(args)