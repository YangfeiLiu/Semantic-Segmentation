import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from albumentations import RandomCrop
import sys
import cv2


root = 'E:\\全国人工智能大赛\\train\\'
# color_map = {"大棚":[0,250,0], "耕地":[0,150,0], "水":[250,250,250], "道路":[100,100,100], 
#              "建筑":[200,200,101], "操场":[250,0,0], "植被":[0,0,250], "背景":[0,0,0]}

color_map = {"背景":0, "道路":2, "耕地":4, "水":1, "草地":5,
             "建筑":3, "林地":6, "裸土":7}


def split_train_val():
    save_image = os.path.join(root, 'image')
    image_list = os.listdir(save_image)
    np.random.shuffle(image_list)
    valid_list = image_list[::4]
    train_list = [x for x in image_list if x not in valid_list]
    with open(os.path.join(root, 'train.txt'), 'w') as f:
        for each in train_list:
            f.write(each.rstrip('.tif') + '\n')
    with open(os.path.join(root, 'valid.txt'), 'w') as f:
        for each in valid_list:
            f.write(each.rstrip('.tif') + '\n')


if __name__ == '__main__':
    split_train_val()
