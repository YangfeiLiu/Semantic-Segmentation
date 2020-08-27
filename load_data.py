from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from torchvision.transforms import transforms
from albumentations import Compose, Flip, RandomResizedCrop, Normalize
import torch


mean = (93.35397203, 111.22929651, 92.32306876)
std  = (24.17318143, 20.92185836, 18.49409081)


class MyData(Dataset):
    def __init__(self, root, phase='train', size=512, channels=3, scale=(0.8, 1.2)):
        self.size = size
        self.channels = channels
        self.scale = scale
        self.image_path = os.path.join(root, 'image')
        self.label_path = os.path.join(root, 'label')
        if phase == 'train':
            self.data_list = open(os.path.join(root, 'train.txt'), 'r').readlines()
        else:
            self.data_list = open(os.path.join(root, 'valid.txt'), 'r').readlines()
    
    def process(self, img, mask):
        transform = Compose([Flip(),
                             RandomResizedCrop(height=self.size, width=self.size, scale=self.scale),
                             Normalize(mean=mean, std=std)])
        augment = transform(image=img, mask=mask)
        return augment['image'], augment['mask']
    
    def merge4img(self, name_list):
        new_img = np.zeros(shape=(self.size * 2, self.size * 2, self.channels))
        new_mask = np.zeros(shape=(self.size() * 2, self.size() * 2), dtype=np.uint8)
        for _ in range(4):
            pass
 
    def __getitem__(self, item):
        img  = Image.open(os.path.join(self.image_path, self.data_list[item].rstrip('\n')))
        mask = Image.open(os.path.join(self.label_path, self.data_list[item].rstrip('\n')))
        img, mask = self.process(np.array(img), np.array(mask))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask)
        img = img.permute(2, 0, 1)
        return img, mask.long()

    def __len__(self):
        return len(self.data_list)


class RoadData(Dataset):
    def __init__(self, root, phase='train', size=512, channels=1, scale=(0.8, 1.2)):
        self.size = size
        self.channels = channels
        self.scale = scale
        self.image_path = os.path.join(root, 'image')
        self.label_path = os.path.join(root, 'label')
        if phase == 'train':
            self.data_list = open(os.path.join(root, 'train.txt'), 'r').readlines()
        else:
            self.data_list = open(os.path.join(root, 'valid.txt'), 'r').readlines()

    def normal(self, img):
        if img.max():
            return img / img.max()
        else:
            return img
    
    def __getitem__(self, item):
        img  = Image.open(os.path.join(self.image_path, self.data_list[item].rstrip('\n')))
        mask = Image.open(os.path.join(self.label_path, self.data_list[item].rstrip('\n')))

        img = np.array(img.convert('L'))
        img = np.expand_dims(self.normal(img), axis=-1)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask)
        img = img.permute(2, 0, 1)
        return img, mask.long()
    
    def __len__(self):
        return len(self.data_list)
