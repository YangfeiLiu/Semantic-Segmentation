import os
from PIL import Image
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ma-user/work/detandclf/vedaseg/')
from vedaseg.utils import MetricMeter

TARGET_H = 1024
TARGET_W = 1024
STRIDE = TARGET_H//2 #推理按照1/2来给，如果是玩训练集需要修改
CLASS_NUM = 2

# ORIN_W = 2748，ORIN_H = 2748，W_SIZE = 3584, H_SIZE = 3584, H_NUM=6
# W_NUM=6 避免魔鬼数字,但是需要修改给进一张原始大小的图片
ORIN_IMAGE_DIR = '/home/ma-user/work/data/glldata/orin_data/cloud/WSISEG/test/image/'
IMAGE_LIST = os.listdir(ORIN_IMAGE_DIR)
IMAGE_PATH = os.path.join(ORIN_IMAGE_DIR, IMAGE_LIST[0])
IMAGE = np.asarray(Image.open(IMAGE_PATH))
ORIN_H, ORIN_W = IMAGE.shape[0], IMAGE.shape[1]
W_SIZE = (ORIN_W//TARGET_W)*TARGET_W if (ORIN_W//TARGET_W==0) else (ORIN_W//TARGET_W+1)*TARGET_W+STRIDE
H_SIZE = (ORIN_H//TARGET_H)*TARGET_H if (ORIN_H//TARGET_H==0) else (ORIN_H//TARGET_H+1)*TARGET_H+STRIDE
H_NUM = (H_SIZE-TARGET_H)//STRIDE+1
W_NUM = (W_SIZE-TARGET_W)//STRIDE+1

def get_big_result(csv_dir, small_image_dir, output_label_dir, 
out_put_label_show_dir, gt_dir):
    # 根据推理的图片拼接成大图同时可视化出来
    location_csv_list = os.listdir(csv_dir)
    metric = MetricMeter(CLASS_NUM)
    for one_csv_name in location_csv_list:
        csv_path = os.path.join(csv_dir, one_csv_name)
        cvs = pd.read_csv(csv_path, header=None)
        out_array = np.zeros((H_SIZE, W_SIZE), dtype=np.uint8)
        for i in range(H_NUM*W_NUM):
            file_name = cvs.iloc[i, 0] #filename
            pos_list = cvs.iloc[i, 1:].values.astype("int") # list
            [topleft_x,topleft_y,buttomright_x,buttomright_y] = pos_list
            small_image_path = os.path.join(small_image_dir, file_name)
            small_array = np.array(Image.open(small_image_path)).astype(np.uint8)
            out_array[topleft_y+STRIDE//2:buttomright_y-STRIDE//2, topleft_x+STRIDE//2:buttomright_x-
            STRIDE//2] = small_array[0+STRIDE//2:TARGET_H-STRIDE//2,0+STRIDE//2:TARGET_W-STRIDE//2]
        h, w = out_array.shape
        print(h, w)
        out_array = out_array[STRIDE//2:h-STRIDE//2, STRIDE//2:w-STRIDE//2] #去除整体边界
        out_array = out_array[0:ORIN_H, 0:ORIN_W] # 去除右下边界
        print(out_array.shape)
        save_name = one_csv_name.replace('csv', 'png')
        out_result = Image.fromarray(out_array)
        out_result_show = Image.fromarray(out_array*255//(CLASS_NUM-1))
        out_result.save(os.path.join(output_label_dir, save_name))
        out_result_show.save(os.path.join(out_put_label_show_dir, save_name))
        gt_path = os.path.join(gt_dir, save_name)
        gt_array = np.array(Image.open(gt_path)).astype(np.uint8)
        metric.reset()
        metric.add(out_array.astype(np.uint8), gt_array.astype(np.uint8))
        mIOU, IOUs = metric.miou()
        print(mIOU, IOUs)



if __name__ == "__main__":
    csv_dir = '/home/ma-user/work/data/glldata/orin_data/cloud/WSISEG/test_data/locations/'
    small_image_dir = '/home/ma-user/work/data/glldata/orin_data/cloud/WSISEG/pre_output/'
    output_label_dir = '/home/ma-user/work/data/glldata/orin_data/cloud/WSISEG/result/label/'
    out_put_label_show_dir = '/home/ma-user/work/data/glldata/orin_data/cloud/WSISEG/result/show/'
    gt_dir = '/home/ma-user/work/data/glldata/orin_data/cloud/WSISEG/test/label/'
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    if not os.path.exists(out_put_label_show_dir):
        os.makedirs(out_put_label_show_dir)
    get_big_result(csv_dir, small_image_dir, output_label_dir, 
    out_put_label_show_dir, gt_dir)