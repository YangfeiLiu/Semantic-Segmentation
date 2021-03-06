from PIL import Image
import os
import matplotlib.pyplot as plt
from albumentations import RandomResizedCrop, RandomCrop
import sys
import cv2
import numpy as np
from tifffile import imread
Image.MAX_IMAGE_PIXELS = 1000000000000000
from tqdm import tqdm
import random


color_map = {"背景": [0, 0, 0], "水田": [0, 200, 0], "水浇地": [150, 250, 0], "旱耕地": [150, 200, 150], "园地": [200, 0, 200],
             "乔木林地": [150, 0, 250], "灌木林地": [150, 150, 250], "天然草地": [250, 200, 0], "人工草地": [200, 200, 0],
             "工业用地": [200, 0, 0], "城市住宅": [250, 0, 150], "村镇住宅": [200, 150, 150], "交通运输": [250, 150, 150],
             "河流": [0, 0, 200], "湖泊": [0, 150, 200], "坑塘": [0, 200, 250]}

gray = [0, 117, 192, 179, 83, 73, 161, 192, 177, 60, 92, 165, 180, 23, 111, 146]


def change(label):
    a = np.zeros_like(label)
    for i in gray:
        a[label == i] = 1
    label = label * a
    for i in gray:
        label[label == i] = gray.index(i)
    return label


def cut_data(img, lab, size=512):
    global cnt
    h, w = img.shape[0], img.shape[1]
    new_w = (w // size) * size if (w // size == 0) else (w // size + 1) * size
    new_h = (h // size) * size if (h // size == 0) else (h // size + 1) * size
    img = np.pad(img, ((0, new_h - h), (0, new_w - w), (0, 0)), mode='constant')
    lab = np.pad(lab, ((0, new_h - h), (0, new_w - w)), mode='constant')
    aug = RandomCrop(height=size, width=size)
    for i in range(0, img.shape[0], size // 2):
        if i + size > img.shape[0]: break
        for j in range(0, img.shape[1], size // 2):
            if j + size > img.shape[1]: break
            img_patch = img[i: i+size, j: j+size, :]
            img_gray = np.array(Image.fromarray(img_patch).convert('L'))
            # if np.sum(img_gray == 0) / (size * size) > 0.5:continue
            lab_patch = lab[i: i+size, j: j+size]
            # if len(np.unique(lab_patch)) == 1: continue
            Image.fromarray(img_patch).save(os.path.join(save_path, 'image', str(cnt) + '.png'))
            Image.fromarray(lab_patch).save(os.path.join(save_path, 'label', str(cnt) + '.png'))
            print(cnt)
            cnt += 1
    for _ in range(500):
        augment = aug(image=img, mask=lab)
        img_patch, lab_patch = augment['image'], augment['mask']
        img_gray = np.array(Image.fromarray(img_patch).convert('L'))
        # print(np.sum(img_gray == 0))
        # if np.sum(img_gray == 0) / (size * size) > 0.5: continue
        # if len(np.unique(lab_patch)) == 1: continue
        Image.fromarray(img_patch).save(os.path.join(save_path, 'image',  str(cnt) + '.png'))
        Image.fromarray(lab_patch).save(os.path.join(save_path,  'label', str(cnt) + '.png'))
        print(cnt)
        cnt += 1


def split_train_val():
    save_image = os.path.join(save_path, 'label512')
    image_list = os.listdir(save_image)
    np.random.shuffle(image_list)
    valid_list = image_list[::5]
    train_list = [x for x in image_list if x not in valid_list]
    with open(os.path.join(save_path, 'train.txt'), 'w') as f:
        for each in train_list:
            f.write(os.path.splitext(each)[0] + '\n')
    with open(os.path.join(save_path, 'valid.txt'), 'w') as f:
        for each in valid_list:
            f.write(os.path.splitext(each)[0] + '\n')


def show(img, lab):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('img')
    plt.subplot(1, 2, 2)
    plt.imshow(lab)
    plt.title('lab')
    plt.show()


def puzzle_image(imgs, labs):
    global cnt
    new_img = np.hstack((np.vstack((imgs[0], imgs[1])), np.vstack((imgs[2], imgs[3]))))
    new_lab = np.hstack((np.vstack((labs[0], labs[1])), np.vstack((labs[2], labs[3]))))
    # show(new_img, new_lab)
    Image.fromarray(new_img).save(save_path + 'image512/' + 'ccf2020_%06d.jpg' % cnt)
    Image.fromarray(new_lab).save(save_path + 'label512/' + 'ccf2020_%06d.png' % cnt)
    cnt += 1
    return new_lab


if __name__ == '__main__':
    save_path = '/workspace/lyf/data/ccf2020/train_data/'
    # split_train_val()
    # exit(0)
    root = '/workspace/lyf/data/ccf2020/train_data/'
    images = os.listdir(os.path.join(root, 'img_train'))
    labels = os.listdir(os.path.join(root, 'lab_train'))
    # items = os.listdir(root)
    num_class = 7
    cnt = 1
    ratio = [0] * num_class
    # labels = [x for x in items if 'class' in x]
    # images = [x for x in items if x not in labels]
    labels.sort()
    images.sort()
    os.makedirs(save_path + 'image512', exist_ok=True)
    os.makedirs(save_path + 'label512', exist_ok=True)
    times = tqdm(range(4))
    a = list(zip(images, labels))
    for k in times:
        times.set_description('time:%d' % k)
        random.shuffle(a)
        imgs = list()
        labs = list()
        each = tqdm(range(len(a)))
        for i in each:
            each.set_description('i=%06d' % i)
            image = np.array(Image.open(os.path.join(root, 'img_train', a[i][0])))
            label = np.array(Image.open(os.path.join(root, 'lab_train', a[i][1])))
            imgs.append(image)
            labs.append(label)
            if i % 4 == 3:
                new_label = puzzle_image(imgs, labs)
                imgs = list()
                labs = list()
                for j in range(num_class):
                    ratio[j] += np.sum(new_label == j)
            # cut_data(image, label, 512)
        # split_train_val()
    with open(save_path + 'weight.txt', 'w') as f:
        ratio_ = [str(it) for it in ratio]
        f.write(' '.join(ratio_))
    print(ratio)
