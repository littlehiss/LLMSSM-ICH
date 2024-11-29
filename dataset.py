import os
import re
import cv2
import numpy
import random
import torch
import json
import numpy as np
import torch.utils.data
# import SimpleITK as sitk
import pandas as pd
# settings total = 100
from torchvision.transforms import transforms
from transformers import CLIPTokenizer
from keep_proportion import get_affine_trans_image
from PIL import Image
from chatgpt_emb import gpt_embeddings
from scipy.ndimage import gaussian_filter

def extract_number(patient_id):
    match = re.search(r'-(\d+)$', patient_id)
    if match:
        return int(match.group(1))
    else:
        return 0

def load_embeddings_from_json(filename):
    with open(filename, 'r') as f:
        json_compatible_dict = json.load(f)
    patients_embeddings = {}
    for patient_name, embedding_list in json_compatible_dict.items():
        patients_embeddings[patient_name] = np.array(embedding_list, dtype=np.float32)

    return patients_embeddings

def screch_excel(file_path, dataset_path):
    df = pd.read_excel(file_path)
    df_dict = df.set_index('new_name').T.to_dict('list')
    patient_names = df['new_name'].tolist()
    all_files = os.listdir(dataset_path)
    patient_files = [f for f in all_files if any(patient_name in f for patient_name in patient_names)]

    data_dict = {}  # 初始化字典
    table_names = ['gender ', 'age ', 'high blood pressure ', 'diabetes ', 'smoke ', 'alcoholic ',
                   'time of onset ', 'GCS ', 'NIHSS ', 'heart rate ', 'potassium ', 'sodium ', 'leukocyte ', 'blood platelet ',
                   'PT ', 'INR ', 'APTT ', 'FIB ', 'TT ', 'D-dimers ']
    flag = 0
    for patient_file in patient_files:
        patient_name = patient_file.split('-')[0]
        if patient_name in df_dict:
            row = df_dict[patient_name]
            text = row[1:]
            words = ''
            for i in range(len(text)):
                words = words + table_names[i] + str(text[i]) + ','
            img_path = f'{dataset_path}/{patient_file}'

            loaded_embeddings = load_embeddings_from_json('/home/xzc/PycharmProjects/ICH-Prognosis/chatgpt/patients_embeddings.json')
            if patient_name in loaded_embeddings:
                embedding = loaded_embeddings[patient_name]
            else:
                print(f"No embedding found for {patient_name}")
            if row[0] == 1:
                flag += 1
            data_dict[patient_name] = {
                'label': row[0],
                'text': embedding,
                'path': img_path
            }
    # print(data_dict)
    print('Patients number:', flag)
    return data_dict

def fft2d(image):
    image = torch.fft.fftshift(image, dim=(-2, -1))
    fft_image = torch.fft.fft2(image)
    fft_image = torch.fft.ifftshift(fft_image, dim=(-2, -1))
    return fft_image

def min_max_normalization(tensor):
    tensor = tensor.float()
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

class dataset2D(torch.utils.data.Dataset):
    def __init__(self, patients_info):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=20, p=0.5),
            transforms.ToTensor(),
        ])
        self.data_list = []
        for info in patients_info:
            img_path = info['path']
            text_info = info['text']
            label = info['label']
            self.data_list.append([img_path, text_info, label])

    def __getitem__(self, index):
        img_path, text_info, label = self.data_list[index]
        patient_name = img_path.split('/')[-1].split('_')[0]

        img = Image.open(img_path).convert('RGB')

        img = np.array(img)

        img_resize = get_affine_trans_image(img, network_input_size=[256, 256])
        img_resize = img_resize["input"]
        img_trans = self.transform(img_resize)

        img_output = min_max_normalization(img_trans)

        text = torch.tensor(text_info, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.long)

        return img_output, text, label, patient_name

    def __len__(self):
        return len(self.data_list)

    def build_index(self):
        self.name_to_index = {self.data_list[i][0].split('/')[1]: i for i in range(len(self.data_list))}