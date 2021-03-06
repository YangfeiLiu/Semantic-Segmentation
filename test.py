from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from models.deeplab import DeepLab
from models.lednet import LEDNet
from models.hrnetv2 import HRnetv2
from models.ocrnet import get_seg_model
from models.dinknet import get_dink_model
from PIL import Image
from albumentations import Resize
from tifffile import imread
Image.MAX_IMAGE_PIXELS = 1e11


class LoadTestData(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_list = os.listdir(root)

    def normal(self, img):
        img = img / 127.5 - 1.0
        return img

    def __getitem__(self, item):
        name = self.img_list[item]
        img = np.array(Image.open(os.path.join(self.root, name)))
        img = Resize(width=512, height=512)(image=img)['image']
        img = self.normal(img)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        return img, name

    def __len__(self):
        return len(self.img_list)


class Infer():
    def __init__(self, in_feats=3, img_mode='RGB', num_classes=5, size=512, stay=300, batch_size=24, modelname='dinknet'):
        backbone = 'resnet101'
        enhance = 'dblock'
        self.test_root = '/workspace/lyf/data/ccf2020/train_data/2016-06/'
        self.in_channels = in_feats
        self.img_mode = img_mode
        self.num_classes = num_classes
        self.pretrain = '/workspace/lyf/data/ccf2020/train_data/save_model/best_deeplab.pth'
        self.save_path = '/workspace/lyf/data/ccf2020/train_data/2016-06-test/'
        os.makedirs(self.save_path, exist_ok=True)
        self.threshold = 0.5
        self.size = size
        self.crop = stay
        self.batch_size = batch_size
        self.modelname = modelname
        if modelname == 'deeplab':
            self.model = DeepLab(in_channels=self.in_channels, backbone=backbone, enhance=enhance, output_stride=16, num_classes=self.num_classes)
        if modelname == 'lednet':
            self.model = LEDNet(num_classes=self.num_classes)
        if modelname == 'hrnetv2':
            self.model = HRnetv2(in_channels=self.in_channels, num_classes=self.num_classes, use_ocr_head=False)
        if modelname == 'ocrnet':
            self.model = get_seg_model(in_channels=self.in_channels, num_classes=self.num_classes, use_ocr_head=True)
        if modelname == 'dinknet':
            self.model = get_dink_model(in_channels=self.in_channels, num_classes=self.num_classes, backbone=backbone)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        try:
            self.model.load_state_dict(torch.load(self.pretrain)['model'])
        except:
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(self.pretrain)['model'])
        self.model.to(self.device)
        self.test_loader = DataLoader(LoadTestData(self.test_root), batch_size=batch_size, num_workers=16)

        self.color_map = np.array([[255, 0, 0], [0, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 255],
                                   [255, 255, 0], [0, 0, 0]], dtype=np.uint8)

    def __call__(self, need_crop=True):
        if need_crop:  # 测试图像是否需要裁剪
            if os.path.isdir(self.test_root):
                # 测试一批图像
                img_list = os.listdir(self.test_root)
                for img_name in tqdm(img_list):
                    img_path = os.path.join(self.test_root, img_name)
                    img = Image.open(img_path)
                    if self.img_mode == 'Gray':
                        img = img.convert('L')
                        img = Image.merge(mode='RGB', bands=(img, img, img))
                        img = np.array(img)
                    else:
                        img = np.array(img)
                    mask = self.testBig(img)
                    color_map = self.color_map[mask]
                    Image.fromarray(color_map).save(os.path.join(self.save_path, img_name))
            else:
                # 测试一副图像
                img = Image.open(self.test_root)
                if self.img_mode == 'Gray':
                    img = img.convert('L')
                    img = Image.merge(mode='RGB', bands=(img, img, img))
                    img = np.array(img)
                else:
                    img = np.array(img)
                mask = self.testBig(img)
                color_map = self.color_map[mask]
                Image.fromarray(color_map).save(self.save_path)
        else:
            self.get_batch()

    def normal(self, img):
        img = img / 255.
        return img

    def testBig(self, img):
        '''采用滑窗的方式测试大图'''
        shape = img.shape
        pad0 = int(np.ceil(shape[0] / self.crop) * self.crop - shape[0])
        pad1 = int(np.ceil(shape[1] / self.crop) * self.crop - shape[1])

        pad = int((self.size - self.crop) / 2)
        padding = ((pad, pad0 + pad), (pad, pad1 + pad), (0, 0))

        new_img = np.pad(img, padding, mode='constant')
        self.mask = np.zeros(shape=(shape[0] + pad0, shape[1] + pad1), dtype=np.uint8)
        bs_patch = list()
        cnt = 0
        self.index = list()
        for i in range(0, new_img.shape[0], self.crop):
            if i > new_img.shape[0] - self.size:
                break
            for j in range(0, new_img.shape[1], self.crop):
                if j > new_img.shape[1] - self.size:
                    break
                patch = new_img[i: i + self.size, j: j + self.size, :]
                patch = self.normal(patch)
                patch = torch.from_numpy(patch).float().unsqueeze(dim=0).permute((0, 3, 1, 2))
                bs_patch.append(patch)
                cnt += 1
                self.index.append((i, j))
                if cnt == self.batch_size:
                    bs_patch_tensor = torch.cat(bs_patch, dim=0)
                    patch_pre = self.testPatch(bs_patch_tensor)
                    patch_pre = patch_pre[:, pad: pad + self.crop, pad: pad + self.crop]
                    self.puzzle(patch_pre)
                    self.index.clear()
                    bs_patch.clear()
                    cnt = 0
        if len(self.index) > 0:
            bs_patch_tensor = torch.cat(bs_patch, dim=0)
            patch_pre = self.testPatch(bs_patch_tensor)
            try:
                patch_pre = patch_pre[:, pad: pad + self.crop, pad: pad + self.crop]
            except:
                patch_pre = patch_pre[pad: pad + self.crop, pad: pad + self.crop]
            self.puzzle(patch_pre)
        mask = self.mask[:shape[0], :shape[1]]
        return mask

    def puzzle(self, pre):
        for k, (i, j) in enumerate(self.index):
            try:
                self.mask[i: i + self.crop, j: j + self.crop] += pre[k, :, :]
            except:
                self.mask[i: i + self.crop, j: j + self.crop] += pre[:, :]

    def testPatch(self, img):
        self.model.eval()
        with torch.no_grad():
            img = img.cuda()
            if self.modelname == 'ocrnet':
                _, prob = self.model(img)
            else:
                prob = self.model(img)
            _, out = torch.max(prob, dim=1)
            pre_array = out.squeeze().cpu().numpy().astype(np.uint8)
            return pre_array

    def get_batch(self):
        '''测试小图'''
        for img, name in tqdm(self.test_loader):
            img = img.to(self.device)
            batch_pre = self.test(img)
            self.save_res(batch_pre, name)

    def test(self, batch):
        with torch.no_grad():
            out = self.model(batch)
            if self.num_classes > 1:
                _, pre = torch.max(out, dim=1)
            else:
                pre = out
                pre[pre >= self.threshold] = 1
                pre[pre <  self.threshold] = 0
            pre = pre.squeeze().cpu().detach().numpy()
            return pre

    def save_res(self, pre, name_list):
        for i, name in enumerate(name_list):
            img = pre[i].astype(np.uint8)
            img *= 255
            img = Image.fromarray(img)
            img.save(os.path.join(self.save_path, name.replace('jpg', 'png')))


if __name__ == '__main__':
    infer = Infer(in_feats=3, num_classes=7, img_mode='Gray', size=512, stay=300, batch_size=24, modelname='deeplab')
    infer(need_crop=True)
