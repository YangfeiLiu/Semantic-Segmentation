from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision.transforms import transforms
from albumentations import Compose, Flip, RandomResizedCrop, Normalize, Resize
import torch
import cv2
import matplotlib.pyplot as plt

## 天智杯
mean = (93.35397203, 111.22929651, 92.32306876)
std  = (24.17318143, 20.92185836, 18.49409081)
maps = [0, 1, 2, 4, 6, 7, 31, 32, 51, 52, 53]


class MyData(Dataset):
    def __init__(self, root='/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/data/segmentation/眼神杯/',
                 phase='train', size=512, channels=3, scale=(0.8, 1.2)):
        self.size = size
        self.channels = channels
        self.scale = scale
        self.image_path = os.path.join(root, 'train_data', 'image')
        self.label_path = os.path.join(root, 'val_data', 'label')
        if phase == 'train':
            self.data_list = open(os.path.join(root, 'train_data', 'train.txt'), 'r').readlines()
        else:
            self.data_list = open(os.path.join(root, 'val_data', 'valid.txt'), 'r').readlines()
    
    def process(self, img, mask):
        transform = Compose([Flip(),
                             RandomResizedCrop(height=self.size, width=self.size, scale=self.scale),
                             ])
        augment = transform(image=img, mask=mask)
        img = augment['image']
        # img = Normalize(mean=mean, std=std)(image=img)['image']
        return img, augment['mask']

    def normal(self, img):
        img = img / 127.5 - 1.0
        return img

    def __getitem__(self, item):
        img  = np.array(Image.open(os.path.join(self.image_path, self.data_list[item].rstrip('\n') + '.tif')))
        mask = np.array(Image.open(os.path.join(self.label_path, self.data_list[item].rstrip('\n') + '.tif')))
        for i in maps:
            mask[mask == i] = maps.index(i)
        mask[mask == 33] = 0
        img, mask = self.process(img, mask)
        # print(mask.max())
        img = self.normal(img)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        img = img.permute(2, 0, 1)
        return img, mask

    def __len__(self):
        return len(self.data_list)


## 二分类数据读取
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
        img  = Image.open(os.path.join(self.image_path, self.data_list[item].rstrip('\n') + '.tif')).convert('L')
        mask = Image.open(os.path.join(self.label_path, self.data_list[item].rstrip('\n') + '.tif'))

        resize = Resize(height=self.size, width=self.size, interpolation=cv2.INTER_NEAREST)(image=np.array(img), mask=np.array(mask))
        img, mask = resize['image'], resize['mask']

        img = np.expand_dims(self.normal(img), axis=-1)

        mask = np.array(mask)
        mask[mask > 0] = 1

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        img = img.permute(2, 0, 1)
        return img, mask
    
    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    CD = MyData()
    data = DataLoader(CD, batch_size=1, shuffle=True)
    for i, j in data:
        i = i.squeeze().data.numpy()
        j = j.squeeze().data.numpy()
        plt.imshow(j)
        plt.show()
