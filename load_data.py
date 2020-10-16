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

## 全国人工智能大赛数据读取
class CompeteData(Dataset):
    def __init__(self, root, size=256, img_size=256, phase='train', channels=3, stride=1, scale=(0.8, 1.2)):
        self.size = size
        self.img_size = img_size
        self.channels = channels
        self.scale = scale
        self.stride = stride
        self.image_path = os.path.join(root, 'image')
        self.label_path = os.path.join(root, 'label')
        if phase == 'train':
            self.data_list = open(os.path.join(root, 'train.txt'), 'r').readlines()
        else:
            self.data_list = open(os.path.join(root, 'valid.txt'), 'r').readlines()
    
    def process_mask(self, mask):
        mask = (mask - 100) / 100
        return mask
    ## 数据增强
    def process(self, img, mask):
        transform = Compose([Flip(),
                             RandomResizedCrop(height=self.size, width=self.size, scale=self.scale),
                             ])
        augment = transform(image=img, mask=mask)
        return augment['image'], augment['mask']
    ## 将几幅小图拼成大图
    def merge_img(self, name_list, img_size):
        nums = sqrt(len(name_list))
        new_img = np.zeros(shape=(img_size * nums, img_size * nums, img_channels))
        new_mask = np.zeros(shape=(img_size * nums, img_size * nums), dtype=np.uint8)
        for i in range(0, nums * img_size, img_size):
            for j in range(0, nums * img_size, img_size):
                item = 0
                img = Image.open(os.path.join(self.image_path, name_list[item].rstrip('\n') + '.tif'))
                if self.channels == 1:
                    img = img.convert('L')
                img  = np.array(img)
                mask = Image.open(os.path.join(self.label_path, name_list[item].rstrip('\n') + '.png'))
                mask = np.array(img)
                new_img[i: i + 1, j: j + 1, :] += img
                new_mask[i: i + 1, j: j + 1, :] += mask
                item += 1
        return new_img, new_mask
    ## 归一化
    def normal(self, img):
        img = img / 127.5 - 1.0
        return img

    def __getitem__(self, item):
        if self.size == self.img_size:
            img  = Image.open(os.path.join(self.image_path, self.data_list[item].rstrip('\n') + '.tif'))
            if self.channels == 1:
                img = img.convert('L')
            mask = Image.open(os.path.join(self.label_path, self.data_list[item].rstrip('\n') + '.png'))   
            img, mask = np.array(img), np.array(mask)
        else:
            num = self.size / self.img_size
            name_list = np.random.choice(a=self.data_list, size=num**2)
            img, mask = self.merge_img(name_list, self.img_size)

        img  = self.normal(img)
        mask = self.process_mask(mask)
        img, mask = self.process(img, mask)
        ## 专门用于hrnetv2训练
        if self.stride != 1:
            mask_size = self.size // self.stride
            mask = Resize(height=mask_size, width=mask_size, interpolation=cv2.INTER_NEAREST)(image=mask)['image']
            # print(mask)
        if self.channels == 1:
            img = np.expand_dims(img, axis=-1)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        img = img.permute(2, 0, 1)
        return img, mask

    def __len__(self):
        return len(self.data_list)


class LoadTestData(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_list = os.listdir(root)
    
    def normal(self, img):
        img = img / 127.5 - 1.0
        return img

    def __getitem__(self, item):
        name = self.img_list[item]
        img = np.array(Image.open(os.path.join(self.root, name)))
        img = self.normal(img)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        return img, name

    def __len__(self):
        return len(self.img_list)


import matplotlib.pyplot as plt
if __name__ == '__main__':
    CD = MyData()
    data = DataLoader(CD, batch_size=1, shuffle=True)
    for i, j in data:
        i = i.squeeze().data.numpy()
        j = j.squeeze().data.numpy()
        plt.imshow(j)
        plt.show()
