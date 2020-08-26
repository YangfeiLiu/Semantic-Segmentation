import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from albumentations import RandomCrop
import sys
import cv2


root = '/media/hp/1500/liuyangfei/tianzhidata/地表分类/训练样本数据/'
# color_map = {"大棚":[0,250,0], "耕地":[0,150,0], "水":[250,250,250], "道路":[100,100,100], 
#              "建筑":[200,200,101], "操场":[250,0,0], "植被":[0,0,250], "背景":[0,0,0]}

color_map = {"背景":[0], "大棚":[147], "耕地":[88], "水":[250], 
             "道路":[100, 101], "建筑":[188, 189], "操场":[74, 75], "植被":[28]}
x = [0, 147, 88, 250, 100, 101, 188, 189, 74, 75, 28]
SIZE = 512
STEP = SIZE // 8


def mergergb():
    """
    有一幅图是rgba四通道，先转为三通道
    """
    img = Image.open(os.path.join(root, 'SampleImg', '地表分类_01.png'))
    b, g, r, a = img.split()
    img = Image.merge('RGB', (b,g,r))
    img.save(os.path.join(root, 'SampleImg', '地表分类_001.png'))


def get_class_ratio():
    """
    获取标签里各类的占比情况
    """
    ratio = np.zeros([8,], dtype=float)
    label_path = os.path.join(root, 'label')
    with open(os.path.join(root, 'ratio.txt'), 'w') as f:
        sum = 0
        labels = [i for i in os.listdir(label_path) if i.endswith('png')]
        for each in labels:
            label = Image.open(os.path.join(label_path, each))
            w, h = label.size
            sum += w*h            
            label_array = np.array(label.convert('L'))
            mask = np.zeros([h, w], dtype=np.uint8)
            for i, item in enumerate(color_map.items()):
                for num in item[1]:
                    ratio[i] += np.sum(label_array == num)
                    mask[label_array == num] += i
            Image.fromarray(mask).save(os.path.join(label_path, each.replace('png', 'jpg')))
            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(label_array)
            # plt.subplot(1,3,2)
            # plt.imshow(np.array(label))
            # plt.subplot(1,3,3)
            # plt.imshow(mask)
            # plt.show()
        ratio /= sum
        for i, item in enumerate(color_map.items()):
            f.write(item[0] + '\t' + str(ratio[i]) + '\n')


def cut_data():
    label_path = os.path.join(root, 'label')
    image_path = os.path.join(root, 'SampleImg')
    labels = [x for x in os.listdir(label_path) if x.endswith('.png')]
    save_label = os.path.join(root, 'train_data1', 'label')
    save_image = os.path.join(root, 'train_data1', 'image')
    os.makedirs(save_label, exist_ok=True)
    os.makedirs(save_image, exist_ok=True)
    cnt = 0
    mean = np.zeros([3,])
    std = np.zeros([3,])
    ratio = np.zeros([8,])
    for index in range(5):
        label_name = labels[index]
        image_name = os.listdir(image_path)[index]
        mask = np.array(Image.open(os.path.join(label_path, label_name)).convert('L'))
        # 数据太脏了，清除掉错误的像素
        a = np.unique(mask)
        for color in a:
            if color not in x:
                mask[mask == color] = 0
        for i, color in enumerate(color_map.values()):
            for c in color:
                mask[mask == c] = i
        img  = np.asarray(Image.open(os.path.join(image_path, image_name)))
        w, h = img.shape[0], img.shape[1]
        pad0 = int(STEP - w % STEP)
        pad1 = int(STEP - h % STEP)
        mask = np.pad(mask, ((0, pad0), (0, pad1)), 'constant')
        img  = np.pad(img, ((0, pad0), (0, pad1), (0, 0)), 'constant')
        # 
        for i in range(0, w, STEP):
            for j in range(0, h, STEP):
                mask_crop = mask[i: i+SIZE, j: j+SIZE]
                img_crop  = img[i: i+SIZE, j: j+SIZE, :]
                if mask_crop.shape != (SIZE, SIZE):
                    continue
                Image.fromarray(mask_crop).save(os.path.join(save_label, '%d.png' % cnt))
                Image.fromarray(img_crop).save(os.path.join(save_image, '%d.png' % cnt))
                print(cnt)
                mean = (mean * cnt + np.mean(img_crop, axis=(0, 1))) / (cnt + 1)
                std  = (std * cnt + np.std(img_crop, axis=(0, 1))) / (cnt + 1)
                for i in range(8):
                    ratio[i] += np.sum(mask_crop == i) / (SIZE*SIZE)
                cnt += 1

        for _ in range(20):
            augment = RandomCrop(height=SIZE, width=SIZE)(image=img, mask=mask)
            img_crop, mask_crop = augment['image'], augment['mask']
            Image.fromarray(mask_crop).save(os.path.join(save_label, '%d.png' % cnt))
            Image.fromarray(img_crop).save(os.path.join(save_image, '%d.png' % cnt))
            print(cnt)
            mean = (mean * cnt + np.mean(img_crop, axis=(0, 1))) / (cnt + 1)
            std  = (std * cnt + np.std(img_crop, axis=(0, 1))) / (cnt + 1)
            for i in range(8):
                ratio[i] += np.sum(mask_crop == i) / (SIZE*SIZE)
            cnt += 1
    print("mean=%s, std=%s, ratio=%s" % (mean, std, ratio))
#     mean=[ 91.595847   110.80768902  91.44284598], std=[21.74898268 18.91127728 16.40839188], ratio=[ 54.95459747   8.59527206 106.53210831 246.61238861  28.59648132
#  103.38300323   6.23130417 100.6644249 ]

def split_train_val():
    save_image = os.path.join(root, 'train_data1', 'image')
    image_list = os.listdir(save_image)
    np.random.shuffle(image_list)
    valid_list = image_list[::4]
    train_list = [x for x in image_list if x not in valid_list]
    with open(os.path.join(root, 'train_data1', 'train.txt'), 'w') as f:
        for each in train_list:
            f.write(each + '\n')
    with open(os.path.join(root, 'train_data1', 'valid.txt'), 'w') as f:
        for each in valid_list:
            f.write(each + '\n')


if __name__ == '__main__':
    # mergergb()
    # cut_data()
    # get_class_ratio()
    split_train_val()
