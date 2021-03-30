from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision.transforms import transforms
from albumentations import Compose, Flip, RandomResizedCrop, RandomBrightnessContrast, ShiftScaleRotate, Resize
import torch
import matplotlib.pyplot as plt
import cv2


class Data(Dataset):
    def __init__(self, root='', phase='train', data_name='potsdam', size=512, img_mode='RGB', n_classes=6, scale=(0.8, 1.2)):
        self.size = size
        self.img_mode = img_mode
        self.n_classes = n_classes
        self.scale = scale
        self.root = root
        self.image_path = os.path.join(root, phase, 'image')
        self.label_path = os.path.join(root, phase, 'label')
        self.phase = phase
        if phase == 'train':
            self.data_list = open(os.path.join(root, phase, 'train.txt'), 'r').readlines()
        elif phase == 'valid':
            self.data_list = open(os.path.join(root, phase, 'valid.txt'), 'r').readlines()
        elif phase == 'test':
            self.data_list = []
        else:
            raise NotImplementedError
        if data_name == 'tianchi':
            self.className = ['bird', 'horse', 'person']
            self.color = [[0, 102, 161], [0, 255, 0], [255, 255, 0], [191, 191, 191], [0, 178, 169], [166, 38, 170],
                          [174, 164, 0], [255, 0, 0], [0, 0, 255], [0, 0, 0]]
            self.gray = [78, 150, 226, 191, 124, 91, 148, 76, 29, 0]
        elif data_name == 'potsdam':
            self.className = ["surfaces", "building", "vegetation", "tree", "car", "background"]
            self.color = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
            self.gray = [255, 29, 179, 150, 226, 76]
        else:
            raise Exception('must give a data name')

    def label2index(self, label):
        new_label = np.zeros_like(label)
        for i in self.gray:
            new_label[label == i] = self.gray.index(i)
        assert new_label.max() <= self.n_classes
        return new_label
    
    def train_process(self, img, mask):
        transform = Compose([Flip(),
                             ShiftScaleRotate(),
                             RandomResizedCrop(height=self.size, width=self.size, scale=self.scale),
                             ])
        augment = transform(image=img, mask=mask)
        return augment['image'], augment['mask']

    def valid_process(self, img, mask):
        augment = Resize(height=self.size, width=self.size, p=1)(image=img, mask=mask)
        return augment['image'], augment['mask']

    def process_img(self, img):
        augment = RandomBrightnessContrast()(image=img)
        return augment['image']

    def normal(self, img):
        img = img / 255.
        return img

    def process_mask(self, mask):
        mask = Resize(height=self.size // 4, width=self.size // 4)(image=mask)['image']
        return mask

    def get_edge(self, label):
        edge = cv2.Canny(label, threshold1=0, threshold2=0)
        return edge

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
        img_name = self.data_list[item].rstrip('\n') + '.png'
        lab_name = self.data_list[item].rstrip('\n') + '.png'
        img = Image.open(os.path.join(self.image_path, img_name))
        if self.img_mode == 'Gray':
            img = img.convert('L')
            img = Image.merge(mode='RGB', bands=(img, img, img))
        img = np.array(img)
        mask = np.array(Image.open(os.path.join(self.label_path, lab_name)).convert('L'))
        mask = self.label2index(mask)
        if self.phase == 'train':
            img, mask = self.train_process(img, mask)
        elif self.phase == 'valid':
            img, mask = self.valid_process(img, mask)
        edge = self.get_edge(mask) / 255.
        # edge = self.process_mask(edge)
        # mask = self.process_mask(mask)
        # img = self.process_img(img)
        img = self.normal(img)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        edge = torch.from_numpy(edge).float()
        return img, mask, edge

    def __len__(self):
        return len(self.data_list)


class HRDataEdge(Dataset):
    def __init__(self, root='', phase='train', size=512, img_mode='RGB', num_classes=6, scale=(0.8, 1.2)):
        self.size = size
        self.img_mode = img_mode
        self.n_classes = num_classes
        self.scale = scale
        self.root = root
        self.image_path = os.path.join(root, 'image')
        self.label_path = os.path.join(root, 'label')
        if phase == 'train':
            self.data_list = open(os.path.join(root, 'train.txt'), 'r').readlines()
        elif phase == 'valid':
            self.data_list = open(os.path.join(root, 'valid.txt'), 'r').readlines()
        elif phase == 'test':
            self.data_list = []
        else:
            raise NotImplementedError

    def __call__(self, data_name):
        if data_name == 'tianchi':
            self.className = ['bird', 'horse', 'person']
            self.color = [[0, 102, 161], [0, 255, 0], [255, 255, 0], [191, 191, 191], [0, 178, 169], [166, 38, 170],
                          [174, 164, 0], [255, 0, 0], [0, 0, 255], [0, 0, 0]]
            self.gray = [78, 150, 226, 191, 124, 91, 148, 76, 29, 0]
        elif data_name == 'potsdam':
            self.className = ["surfaces", "building", "vegetation", "tree", "car", "background"]
            self.color = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
            self.gray = [255, 29, 179, 150, 226, 76]
        else:
            raise Exception('must give a data name')

    def process(self, img, mask):
        transform = Compose([Flip(),
                             ShiftScaleRotate(),
                             RandomResizedCrop(height=self.size, width=self.size, scale=self.scale),
                             ])
        augment = transform(image=img, mask=mask)
        return augment['image'], augment['mask']

    def normal(self, img):
        img = img / 255.
        return img

    def change_label(self, label):
        gray = [255, 29, 179, 150, 226, 76]
        for i in gray:
            label[label == i] = gray.index(i)
        assert label.max() <= self.n_classes
        return label

    def process_mask(self, mask):
        mask = Resize(height=self.size // 4, width=self.size // 4)(image=mask)['image']
        return mask

    def show(self, img, lab, edge, img_name, lab_name):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(img_name)
        plt.subplot(1, 3, 2)
        plt.imshow(lab)
        plt.title(lab_name)
        plt.subplot(1, 3, 3)
        plt.imshow(edge)
        plt.title(lab_name)
        plt.show()

    def get_edge(self, label):
        edge = cv2.Canny(label, threshold1=0, threshold2=0)
        return edge

    def __getitem__(self, item):
        img_name = self.data_list[item].rstrip('\n') + '.png'
        lab_name = self.data_list[item].rstrip('\n') + '.png'
        img = Image.open(os.path.join(self.image_path, img_name))
        if self.img_mode == 'Gray':
            img = img.convert('L')
            img = Image.merge(mode='RGB', bands=(img, img, img))
        img = np.array(img)
        mask = np.array(Image.open(os.path.join(self.label_path, lab_name)).convert('L'))
        mask = self.change_label(mask)
        img, mask = self.process(img, mask)
        # print(np.unique(mask))
        edge = self.get_edge(mask) // 255
        img = self.normal(img)
        # mask = self.process_mask(mask)
        # edge = self.process_mask(edge)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        edge = torch.from_numpy(edge).float()
        return img, mask, edge

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    CD = HRDataEdge(root='/workspace/2/data/vaihingen/', phase='test')
    data = DataLoader(CD, batch_size=1, shuffle=True, num_workers=4)
    for i, j, k, in data:
        print('-------')
        i = i.squeeze().data.numpy()
        j = j.squeeze().data.numpy()
        # plt.imshow(j)
        # plt.show()
