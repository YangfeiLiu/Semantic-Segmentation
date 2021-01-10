import numpy as np
import tqdm
import glob
from tifffile import imread, imwrite
import os
import math


def auto_tune(I, percent):
    row, col = I.shape()
    I_sort = np.sort(I.flatten()).tolist()
    I_out = np.zeros_like(I)
    if percent == 0:
        min_v = min(I_sort)
        max_v = max(I_sort)
    else:
        min_v = I_sort[math.floor(row * col * percent)]
        max_v = I_sort[math.floor(row * col * (1 - percent))]
    I_out[I < min_v] = 0
    I_out[I > max_v] = 1
    I_out[I != 0 & I != 1] = (I[I != 0 & I != 1] - min_v) / (max_v - min_v)
    return I_out

def auto_tune1(I, Max, Min):
    I_out = np.zeros_like(I)
    I_out[I < Min] = 0
    I_out[I > Max] = 1
    I_out[I != 0 & I != 1] = (I[I != 0 & I != 1] - Min) / (Max - Min)
    return I_out

def auto_tone(img, percent=0.001):
    """
    自动色调
    """
    img = img / 255.
    img = auto_tune(img, percent)
    return img


def findMaxMin(I, percent):
    row, col = I.shape()
    I_sort = np.sort(I.flatten()).tolist()
    if percent == 0:
        min_v = min(I_sort)
        max_v = max(I_sort)
    else:
        min_v = I_sort[math.floor(row * col * percent)]
        max_v = I_sort[math.floor(row * col * (1 - percent))]
    return min_v, max_v
 

def auto_contrast(img, percent=0.001):
    img = img / 255.
    Min, Max = findMaxMin(img, percent)
    img = auto_tune1(img, Max, Min)
    return img


def linear_stretch(img, min_value=0, max_value=65535, ratio=2):
    """
    线性拉伸，处理16bit数据
    """
    high_value = np.percentile(img, 100 - ratio)
    low_value = np.percentile(img, ratio)
    new_img = min_value + ((img - low_value) / (high_value - low_value)) * (max_value - min_value)
    new_img[new_img < min_value] = min_value
    new_img[new_img > max_value] = max_value
    new_img = new_img.astype(np.uint16)
    return new_img


def process_img(image_path, ratio=5):
    image = imread(image_path)[:, :, 1:]
    image = image[:, :, ::-1]
    new_image = linear_stretch(image, ratio=ratio)
    image_path = image_path[:-5] + '_%d%%_' % ratio + '16bit.tiff'
    imwrite(image_path, new_image)


if __name__ == '__main__':
    root = '/media/hb/LIU/GeoData/*/*/*/*MSS[1|2].tiff'
    tbar = tqdm.tqdm(glob.glob(root))
    for path in tbar:
        tbar.set_description('img_path:%s\n' % path)
        process_img(path)
