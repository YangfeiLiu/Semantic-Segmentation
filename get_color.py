from PIL import Image
import numpy as np
import os
from tifffile import imread
import matplotlib.pyplot as plt
from shutil import copy
from utils.metrics import MetricMeter
root = '/workspace/lyf/data/vaihingen/valid_result/'
save_path = root
from tqdm import tqdm
import cv2
os.makedirs(save_path, exist_ok=True)
classes = ['耕地', '林地', '草地', '道路', '城镇', '农村', '工业', '构筑物', '水域', '裸地']
color_map = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]], dtype=np.uint8)


def fun():
    for dir in os.listdir(root):
        if dir != 'hrnetv2_edge': continue
        labels = [x for x in os.listdir(os.path.join(root, dir)) if x.endswith('png')]
        for i in labels:
            label = np.array(Image.open(os.path.join(root, dir, i)))
            # print(np.unique(label))
            try:
                color_label = color_map[label]
                Image.fromarray(color_label).save(os.path.join(save_path, dir, i))
            except:
                pass

def to_gray(path='/workspace/lyf/data/vaihingen/train/image/'):
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i), 0)
        cv2.imwrite(os.path.join('/workspace/lyf/data/vaihingen/train/gray/', i), img)
# to_gray()
def fun1(path):
    # images = [x for x in os.listdir(path) if x.endswith('tif')]
    images = open('/workspace/lyf/data/potsdam/train/valid.txt').readlines()
    for i in images:
        i = i.rstrip('\n') + '.png'
        copy(os.path.join(path, i), '/workspace/lyf/data/potsdam/valid_result/valid_label/')
        # image = np.array(Image.open(os.path.join(path, i)))
        # Image.fromarray(image).save(os.path.join('/workspace/lyf/data/vaihingen/valid_result/valid_image/', i))


def fun2():
    labels = os.listdir(os.path.join(root, 'valid_image'))
    for i in labels:
        plt.figure(figsize=(50, 10), frameon=False)
        for j, dir in enumerate(os.listdir(root)):
            label = np.array(Image.open(os.path.join(root, dir, i)))
            plt.subplot(1, 5, j + 1)
            plt.imshow(label)
            title = dir.split('_')
            plt.title(title[0] + '\n' + title[1])
        plt.savefig(os.path.join('/workspace/lyf/data/vaihingen/visib', i))
# fun1('/workspace/lyf/data/potsdam/train/label')
# fun()
import cv2
def get_edge(path):
    labels = os.listdir(path)
    for i in labels:
        label = cv2.imread(os.path.join(path, i), 0)
        edge = cv2.Canny(label, threshold1=0, threshold2=0)
        cv2.imwrite(os.path.join('/workspace/lyf/data/vaihingen/train/edge', i), edge)

# get_edge('/workspace/lyf/data/vaihingen/train/label/')
# fun2()

# gray = [255, 29, 178, 149, 225, 76]
gray = [78, 149, 225, 191, 123, 91, 148, 76, 29, 0]


def change(label):
    a = np.zeros_like(label)
    for i in gray:
        a[label == i] = 1
    label = label * a
    for i in gray:
        a[label == i] = gray.index(i)
    return a

label_path = '/workspace/lyf/data/tianchi/train_result/train_label/'
pred_path = '/workspace/lyf/data/tianchi/train_result/resnetcbam101_aspp/'

def mertric():
    mer = MetricMeter(nclasses=10)
    mer.reset()
    label_list = os.listdir(label_path)
    for name in tqdm(label_list):
        label = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
        label = label - 1
        pred = cv2.imread(os.path.join(pred_path, name.replace('png', 'tif')), cv2.IMREAD_GRAYSCALE)
        # print(np.unique(pred), np.unique(label))
        # label = change(label)
        # pred = change(pred)
        try:
            mer.add(pred, label)
        except:
            print(name)
            print(pred.shape, label.shape)
    miou, ious = mer.miou()
    fwiou = mer.fw_iou()
    pa = mer.pixel_accuracy()
    mpa =mer.pixel_accuracy_class()
    print("miou=%.4f,fwiou=%.4f,pa=%.4f,mpa=%.4f, ious=%s" % (miou, fwiou, pa, mpa, str(ious)))
mertric()

