from PIL import Image
import os
import matplotlib.pyplot as plt
from albumentations import RandomCrop
import numpy as np
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = 1000000000000000


def get_gray():
    rgb = []
    for pix in rgb:
        pix = np.array(pix).reshape((1, 1, 3)).astype(np.uint8)
        gray = np.array(Image.fromarray(pix).convert('L'))
        print(gray)


def gray_label(label):
    gray = [255, 29, 179, 150, 226, 76]
    for i in gray:
        label[label == i] = gray.index(i)
    assert label.max() < num_class
    return label


def cut_img_and_lab(img, lab, size=512, step=512, aug_ratio=0.3):
    if len(lab.shape) > 2 and lab.shape[-1] > 1:
        raise Exception("lab must be gray")
    global cnt
    aug = RandomCrop(height=size, width=size)
    row_flag = False
    num = 1
    for i in range(0, img.shape[0], step):
        if i + size > img.shape[0]:
            if row_flag: break
            i = img.shape[0] - size
            row_flag = True
        col_flag = False
        for j in range(0, img.shape[1], step):
            if j + size > img.shape[1]:
                if col_flag: break
                j = img.shape[1] - size
                col_flag = True

            img_patch = img[i: i+size, j: j+size, :]
            lab_patch = lab[i: i+size, j: j+size]
            Image.fromarray(img_patch).save(os.path.join(save_path, phase, 'image', str(cnt) + '.png'))
            Image.fromarray(lab_patch).save(os.path.join(save_path, phase, 'label', str(cnt) + '.png'))
            cnt += 1
            num += 1
            if phase == 'train':
                for c in range(num_class):
                    ratio[c] += np.sum(lab_patch == c)
    for _ in range(int(num * aug_ratio)):
        augment = aug(image=img, mask=lab)
        img_patch, lab_patch = augment['image'], augment['mask']
        Image.fromarray(img_patch).save(os.path.join(save_path, phase, 'image',  str(cnt) + '.png'))
        Image.fromarray(lab_patch).save(os.path.join(save_path,  phase, 'label', str(cnt) + '.png'))
        cnt += 1
        if phase == 'train':
            for c in range(num_class):
                ratio[c] += np.sum(lab_patch == c)


def split_train_val(ratio=5):
    save_image = os.path.join(save_path, 'label')
    image_list = os.listdir(save_image)
    np.random.shuffle(image_list)
    valid_list = image_list[::ratio]
    train_list = [x for x in image_list if x not in valid_list]
    with open(os.path.join(save_path, 'train.txt'), 'w') as f:
        for each in train_list:
            f.write(os.path.splitext(each)[0] + '\n')
    with open(os.path.join(save_path, 'valid.txt'), 'w') as f:
        for each in valid_list:
            f.write(os.path.splitext(each)[0] + '\n')


def generate_txt(phase):
    img_list = os.listdir(os.path.join(save_path, phase, 'image'))
    # if os.path.exists(os.path.join(save_path, 'train.txt')):
    #     train_list = [x.rstrip('\n') for x in open(os.path.join(save_path, 'train.txt'), 'r').readlines()]
    #     test_list = [x for x in img_list if x.rstrip('.png') not in train_list]
    #     img_list = test_list
    with open(os.path.join(save_path, phase, phase + '.txt'), 'w') as f:
        for each in img_list:
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


def get_weight():
    weight = [0] * num_class
    for item in tqdm(os.listdir(os.path.join(save_path, phase, 'label'))):
        lab = np.array(Image.open(os.path.join(save_path, phase, 'label', item)))
        for c in range(num_class):
            weight[c] += np.sum(lab == c)
    return weight


def puzzle_image(imgs, labs):
    global cnt
    new_img = np.hstack((np.vstack((imgs[0], imgs[1])), np.vstack((imgs[2], imgs[3]))))
    new_lab = np.hstack((np.vstack((labs[0], labs[1])), np.vstack((labs[2], labs[3]))))
    Image.fromarray(new_img).save(save_path + 'image512/' + 'ccf2020_%06d.jpg' % cnt)
    Image.fromarray(new_lab).save(save_path + 'label512/' + 'ccf2020_%06d.png' % cnt)
    cnt += 1
    return new_lab


if __name__ == '__main__':
    num_class = 6
    cnt = 1
    ratio = [0] * num_class
    size = 512
    step = 384
    data_path = '/workspace/2/data/potsdam/'
    save_path = '/workspace/2/data/potsdam/'
    phase = 'valid'
    # for i in os.listdir(os.path.join(data_path, 'labels', phase)):
    #     lab = np.array(Image.open(os.path.join(data_path, 'labels', phase, i)).convert('L'))
    #     print(np.unique(lab), i)
    # print(get_weight())
    # generate_txt(phase)
    # exit(0)
    train_save_path = os.path.join(save_path, 'train')
    valid_save_path = os.path.join(save_path, 'valid')
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(valid_save_path, exist_ok=True)
    images = os.listdir(os.path.join(data_path, 'images', phase))
    labels = os.listdir(os.path.join(data_path, 'labels', phase))
    labels.sort()
    images.sort()

    os.makedirs(os.path.join(train_save_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(train_save_path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(valid_save_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(valid_save_path, 'label'), exist_ok=True)
    num_of_patches = tqdm(zip(images, labels))
    for img_name, lab_name in num_of_patches:
        num_of_patches.set_description('number:%d' % cnt)
        image = np.array(Image.open(os.path.join(data_path, 'images', phase, img_name)))
        label = np.array(Image.open(os.path.join(data_path, 'labels', phase, lab_name)).convert('L'))
        #  将标签转为灰度
        label = gray_label(label)
        cut_img_and_lab(image, label, size, step)
    #  生成索引
    print(ratio)
    generate_txt(phase)
    #  记录训练集各类权重
    if phase == 'train':
        with open(os.path.join(train_save_path, 'weight.txt'), 'w') as f:
            ratio_ = [str(it) for it in ratio]
            f.write(' '.join(ratio_))

