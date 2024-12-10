import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MLDataste(Dataset):
    def __init__(self, img_dir, normal_transform=None, aug_transform=None, mode='test'):
        self.img_dir = img_dir
        self.img_path = []
        self.labels = []
        self.index_mapping = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3,
                         "Neutral": 4, "Sad": 5, "Surprise": 6} 
        self.mode = mode
        if self.mode == 'train':
            for emo_class in self.index_mapping.keys():
                class_folder_path = os.path.join(img_dir, emo_class)
                for fname in sorted(os.listdir(class_folder_path)):
                    self.img_path.append(os.path.join(class_folder_path, fname))
                    self.labels.append(self.index_mapping[emo_class])
        elif self.mode == 'test':
                for fname in sorted(os.listdir(img_dir)):
                    self.img_path.append(os.path.join(img_dir, fname))
        else:
            print('Error: unknown dataset mode')
        self.normal_transform = normal_transform
        self.aug_transform = aug_transform

    def __len__(self):
        if self.aug_transform:
            return len(self.img_path) * 2
        else:
            return len(self.img_path)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.mode == 'train':
            flag = 0
            if idx >= len(self.img_path):
                idx %= len(self.img_path)
                flag = 1
            img_name = self.img_path[idx]
            image = Image.open(img_name)
            label = self.labels[idx]
            if self.aug_transform and flag:
                image = self.aug_transform(image)
            elif self.normal_transform:
                image = self.normal_transform(image)
            return image, label
        
        elif self.mode == 'test':
            img_name = self.img_path[idx]
            image = Image.open(img_name)
            if self.normal_transform:
                image = self.normal_transform(image)
            return image, img_name
        return None