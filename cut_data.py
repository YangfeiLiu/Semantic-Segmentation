'''
这个代码是为了对遥感数据集做一个切分，为了区别训练集与验证集，训练的时候我们期望
使用一个较小的步长，然后对标签进行判断，标签矩阵中有效值小于1/5的认为值是无效的，标准可以更换。
切分完成后检查一下image和label是否都可以正常打开(dataloader)
'''

import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import os
import shutil
Image.MAX_IMAGE_PIXELS = 1000000000000

def image_resize_save_mask(path):
    #1 打开图片
    img = Image.open(path)
    img = np.asarray(img)
    mask = img[:,:,-1]

    #2 缩放
    cimg = cv.resize(img,None,fx=0.1,fy=0.1)


    #3 保存
    ##3.1 BGR2RGB(PIL通道读取与cv2相反)
    cimg = cv.cvtColor(cimg,cv.COLOR_RGB2BGR)
    ##3.2 解析保存路径
    save_dir = r"./vis/"
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    root_path,png_name = os.path.split(path)
    filename,filetype = os.path.splitext(png_name)

    vis_mask_name = os.path.join(save_dir,filename+"_mask_vis"+filetype)
    mask_name = os.path.join(save_dir,filename+"_mask"+filetype)
    png_name = os.path.join(save_dir,filename+"_vis"+filetype)
    
    # cv.imwrite(png_name,cimg,[int(cv.IMWRITE_JPEG_QUALITY),100])
    cv.imwrite(png_name,cimg)
    cv.imwrite(vis_mask_name,mask)
    cv.imwrite(mask_name,mask//255)



def cut_image_save(orin_image_dir, orin_label_dir, orin_label_show_dir, save_dir, 
                        target_size=(1024, 1024), is_train=True, is_show=True):
    save_image_dir = os.path.join(save_dir,"JPEGImages")
    save_label_dir = os.path.join(save_dir,"EncodeSegmentationClass")
    save_csv_dir = os.path.join(save_dir, "locations")
    if not os.path.isdir(save_image_dir): os.makedirs(save_image_dir)  # 保存的label和image目录不存在就创建
    if not os.path.isdir(save_label_dir): os.makedirs(save_label_dir)
    if not os.path.exists(save_csv_dir): os.makedirs(save_csv_dir)
    if is_show:
        save_label_show_dir = os.path.join(save_dir, "show")
        if not os.path.exists(save_label_show_dir): os.makedirs(save_label_show_dir)

    target_size=target_size
    if is_train:
        stride = target_size[0]//8
    else:
        stride = target_size[0]//2
    
    is_train = 1 if is_train else 0

    image_names = os.listdir(orin_image_dir)
    for filename in image_names:
        basename,filetype = os.path.splitext(filename)
        image_path = os.path.join(orin_image_dir, filename)
        label_path = os.path.join(orin_label_dir, filename)
        label_show_path = os.path.join(orin_label_show_dir, filename)
        image = np.asarray(Image.open(image_path))
        label = np.asarray(Image.open(label_path))
        label_show = np.asarray(Image.open(label_show_path))
        # image = cv.imread(image_path,cv.IMREAD_UNCHANGED)
        # label = cv.imread(label_path,cv.IMREAD_GRAYSCALE)
        cnt = 0
        csv_pos_list = []
        

        # 填充外边界至步长整数倍
        target_w,target_h = target_size
        h,w = image.shape[0],image.shape[1]
        print('原始大小：', w, h)
        new_w = (w//target_w)*target_w if (w//target_w == 0) else (w//target_w+1)*target_w
        new_h = (h//target_h)*target_h if (h//target_h == 0) else (h//target_h+1)*target_h
        print('填充值整数倍：', new_w, new_h) # 右下方填充
        image = cv.copyMakeBorder(image,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)
        label = cv.copyMakeBorder(label,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)
        label_show = cv.copyMakeBorder(label_show,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)

        # 填充1/2 stride长度的外边框,四周填充，总共填充
        h,w = image.shape[0],image.shape[1]
        new_w,new_h = w + stride,h + stride # 四周填充
        image = cv.copyMakeBorder(image,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,0)
        label = cv.copyMakeBorder(label,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,0)
        label_show = cv.copyMakeBorder(label_show,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,0)
        print('填充外边框：', new_w, new_h)

        def crop(cnt,crop_image,crop_label, crop_label_show, is_show=is_show):
            image_name = os.path.join(save_image_dir,basename+"_"+str(cnt)+".png")
            label_name = os.path.join(save_label_dir,basename+"_"+str(cnt)+".png")
            cv.imwrite(image_name,crop_image)
            cv.imwrite(label_name,crop_label)
            if is_show:
                label_show_name = os.path.join(save_label_show_dir, basename+"_"+str(cnt)+".png")
                cv.imwrite(label_show_name, crop_label_show)
                
            
        h,w = image.shape[0],image.shape[1]
        for i in range((new_w-target_w)//stride+1):
            for j in range((new_h-target_h)//stride+1):
                topleft_x = i*stride
                topleft_y = j*stride
                crop_image = image[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                crop_label = label[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                crop_label_show = label_show[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]

                if crop_image.shape[:2]!=(target_h,target_h):  # 康康有没有妖魔鬼怪
                    print(topleft_x,topleft_y,crop_image.shape)
                # 有效值小于五分之一不要，这里除了0就1所以可以这么算霾2000，雾5000
                if np.sum(crop_label) < is_train*target_h*target_w//2000:  
                    print(np.sum(crop_label), is_train*target_h*target_w//2000)
                    pass
                else:
                    crop(cnt,crop_image,crop_label, crop_label_show)
                    csv_pos_list.append([basename+"_"+str(cnt)+".png",
                    topleft_x,topleft_y,topleft_x+target_w,topleft_y+target_h])
                    cnt += 1
        csv_pos_list = pd.DataFrame(csv_pos_list)  # 把坐标都保存起来后面方便一一对应，主要是用于验证集。
        csv_pos_list.to_csv(os.path.join(save_csv_dir,basename+".csv"),header=None,index=None)


def check_data_label(train_save_dir, test_save_dir):
    train_data_dir = os.path.join(train_save_dir, 'JPEGImages/')
    train_label_dir = os.path.join(train_save_dir, 'EncodeSegmentationClass/')
    all_image_names = os.listdir(train_data_dir)
    for image_name in all_image_names:
        image_path = os.path.join(train_data_dir, image_name)
        label_path = os.path.join(train_label_dir, image_name)
        try:
            img = cv.imread(image_path).astype(np.float32)
            mask = np.array(Image.open(label_path), dtype=np.float32)
        except:
            print(image_name)


def divid_train_val_by_day(data_root_dir):
    # 把12号的数据作为验证集其他的都是训练集
    all_image_dir = os.path.join(data_root_dir, 'image/')
    all_label_dir = os.path.join(data_root_dir, 'label/')
    all_label_show_dir = os.path.join(data_root_dir, 'label_show/')
    orin_train_dir = os.path.join(data_root_dir, 'train/')
    orin_test_dir = os.path.join(data_root_dir, 'test/')
    orin_train_image = os.path.join(orin_train_dir, 'image/')
    if not os.path.exists(orin_train_image): os.makedirs(orin_train_image)
    orin_train_label = os.path.join(orin_train_dir, 'label/')
    if not os.path.exists(orin_train_label): os.makedirs(orin_train_label)
    orin_train_labelshow = os.path.join(orin_train_dir, 'label_show/')
    if not os.path.exists(orin_train_labelshow): os.makedirs(orin_train_labelshow)
    orin_test_image = os.path.join(orin_test_dir, 'image/')
    if not os.path.exists(orin_test_image): os.makedirs(orin_test_image)
    orin_test_label = os.path.join(orin_test_dir, 'label/')
    if not os.path.exists(orin_test_label): os.makedirs(orin_test_label)
    orin_test_labelshow = os.path.join(orin_test_dir, 'label_show/')
    if not os.path.exists(orin_test_labelshow): os.makedirs(orin_test_labelshow)
    image_name_list = os.listdir(all_image_dir)
    for image_name in image_name_list:
        image_path = os.path.join(all_image_dir, image_name)
        label_path = os.path.join(all_label_dir, image_name)
        label_show_path = os.path.join(all_label_show_dir, image_name)
        if '20200324' in image_name:
            image_save_path = os.path.join(orin_test_image, image_name)
            label_save_path = os.path.join(orin_test_label, image_name)
            label_show_save_path = os.path.join(orin_test_labelshow, image_name)
        else:
            image_save_path = os.path.join(orin_train_image, image_name)
            label_save_path = os.path.join(orin_train_label, image_name)
            label_show_save_path = os.path.join(orin_train_labelshow, image_name)
        shutil.copyfile(image_path, image_save_path)
        shutil.copyfile(label_path, label_save_path)
        shutil.copyfile(label_show_path, label_show_save_path)


if __name__ == "__main__":
    train_orin_dir = '/home/ma-user/work/data/glldata/orin_data/all_data/train/'
    test_orin_dir = '/home/ma-user/work/data/glldata/orin_data/all_data/test/'
    train_save_dir = '/home/ma-user/work/data/glldata/orin_data/all_data/train_data/'
    test_save_dir = '/home/ma-user/work/data/glldata/orin_data/all_data/test_data/'
    if not os.path.exists(train_save_dir): os.makedirs(train_save_dir)
    if not os.path.exists(test_save_dir): os.makedirs(test_save_dir)
    train_image_dir = os.path.join(train_orin_dir, 'image/')
    train_label_dir = os.path.join(train_orin_dir, 'label/')
    train_label_show_dir = os.path.join(train_orin_dir, 'label_show/')
    test_image_dir = os.path.join(test_orin_dir, 'image/')
    test_label_dir = os.path.join(test_orin_dir, 'label/')
    test_label_show_dir = os.path.join(test_orin_dir, 'label_show/')
    data_root_dir = '/home/ma-user/work/data/glldata/orin_data/all_data/'
    divid_train_val_by_day(data_root_dir)
    cut_image_save(train_image_dir, train_label_dir, train_label_show_dir, 
    train_save_dir, target_size=(1024, 1024), is_train=True, is_show=False)
    cut_image_save(test_image_dir, test_label_dir, test_label_show_dir, 
    test_save_dir, target_size=(1024, 1024), is_train=False, is_show=False)
    # check_data_label(train_save_dir, test_save_dir)