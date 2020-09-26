import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from albumentations import RandomResizedCrop
import sys
import cv2


root = '/media/hp/1500/liuyangfei/全国人工智能大赛/train/'
# color_map = {"大棚":[0,250,0], "耕地":[0,150,0], "水":[250,250,250], "道路":[100,100,100], 
#              "建筑":[200,200,101], "操场":[250,0,0], "植被":[0,0,250], "背景":[0,0,0]}

color_map = {"背景":0, "道路":2, "耕地":4, "水":1, "草地":5,
             "建筑":3, "林地":6, "裸土":7}


def cut_data(img, lab, size=512):
    h, w = img.shape[0], img.shape[1]
    new_w = (w // size) * size if (w // size == 0) else (w // size + 1) * size
    new_h = (h // size) * size if (h // size == 0) else (h // size + 1) * size
    img = np.pad(img, ((0, new_h - h), (0, new_w - w), (0, 0)), mode='constant')
    lab = np.pad(lab, ((0, new_h - h), (0, new_w - w)), mode='constant')
    cnt = 1
    aug = RandomResizedCrop(height=size, width=size, scale=(0.8, 1.2))
    for i in range(0, img.shape[0], size // 4):
        if i + size > img.shape[0]: break
        for j in range(0, img.shape[1], size // 4):
            if j + size > img.shape[1]: break
            img_patch = img[i: i+size, j: j+size, :]
            lab_patch = lab[i: i+size, j: j+size]
            Image.fromarray(img_patch).save(os.path.join(save_path, str(cnt) + '.tif'))
            Image.fromarray(lab_patch).save(os.path.join(save_path, str(cnt) + '.tif'))
            print(cnt)
            cnt += 1
    for _ in range(2000):
        augment = aug(image=img, mask=lab)
        img_patch, lab_patch = augment['image'], augment['mask']
        Image.fromarray(img_patch).save(os.path.join(save_path, str(cnt) + '.tif'))
        Image.fromarray(lab_patch).save(os.path.join(save_path, str(cnt) + '.tif'))
        print(cnt)
        cnt += 1


def split_train_val():
    save_image = os.path.join(root, 'label')
    image_list = os.listdir(save_image)
    np.random.shuffle(image_list)
    valid_list = image_list[::4]
    train_list = [x for x in image_list if x not in valid_list]
    with open(os.path.join(root, 'train.txt'), 'w') as f:
        for each in train_list:
            f.write(os.path.splitext(each)[0] + '\n')
    with open(os.path.join(root, 'valid.txt'), 'w') as f:
        for each in valid_list:
            f.write(os.path.splitext(each)[0] + '\n')


if __name__ == '__main__':
    save_path = ''
    split_train_val()
