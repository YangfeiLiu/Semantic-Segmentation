import numpy as np
import tqdm
import glob
from tifffile import imread, imwrite
import os


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
