from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from albumentations import Compose, Flip, RandomResizedCrop
import torch
import matplotlib.pyplot as plt


class MyData(Dataset):
    def __init__(self, root='', phase='train', size=512, img_mode='RGB', n_classes=1, scale=(0.8, 1.2)):
        self.size = size
        self.img_mode = img_mode
        self.n_classes = n_classes
        self.scale = scale
        self.image_path = os.path.join(root, 'image512')
        self.label_path = os.path.join(root, 'label512')
        if phase == 'train':
            self.data_list = open(os.path.join(root, 'train512.txt'), 'r').readlines()
        else:
            self.data_list = open(os.path.join(root, 'valid512.txt'), 'r').readlines()
    
    def process(self, img, mask):
        transform = Compose([Flip(),
                             RandomResizedCrop(height=self.size, width=self.size, scale=self.scale),
                             ])
        augment = transform(image=img, mask=mask)
        return augment['image'], augment['mask']

    def normal(self, img):
        img = img / 255.
        return img

    def process_mask(self, mask):
        mask = mask / 100 - 1
        return mask

    def show(self, img, lab, img_name, lab_name):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(img_name)
        plt.subplot(1, 2, 2)
        plt.imshow(lab)
        plt.title(lab_name)
        plt.show()

    def __getitem__(self, item):
        img_name = self.data_list[item].rstrip('\n') + '.jpg'
        lab_name = self.data_list[item].rstrip('\n') + '.png'
        img  = Image.open(os.path.join(self.image_path, img_name))
        if self.img_mode == 'Gray':
            img = img.convert('L')
            img = Image.merge(mode='RGB', bands=(img, img, img))
        img = np.array(img)
        mask = np.array(Image.open(os.path.join(self.label_path, lab_name)))
        # mask = self.process_mask(mask)
        img, mask = self.process(img, mask)
        img = self.normal(img)
        # self.show(img, mask, img_name, lab_name)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask)
        if self.n_classes == 1:
            mask = mask.float()
        else:
            mask = mask.long()
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
