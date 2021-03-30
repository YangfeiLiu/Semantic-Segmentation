from torch.utils.data import DataLoader, Dataset
from ReadData import HRDataEdge
from train import Trainer
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from config.load_config import LoadConfig
from PIL import Image
from torch.nn import functional as F
from tifffile import imwrite
Image.MAX_IMAGE_PIXELS = 1e11


class LoadTestData(HRDataEdge):
    def __init__(self, root, phase='test'):
        super().__init__(root, phase)
        self.root = root
        self.img_list = os.listdir(root)

    def __getitem__(self, item):
        name = self.img_list[item]
        img = np.array(Image.open(os.path.join(self.root, name)))
        try:
            img = self.normal(img)
        except:
            print(name)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        return img, name

    def __len__(self):
        return len(self.img_list)


class Infer(Trainer):
    """
    推理类继承了训练类，可以省去模型的重复定义和一些参数的设置
    """
    def __init__(self, config_path):
        super().__init__(config_path)
        self.test_config = LoadConfig(config_path=config_path).test_config()
        os.makedirs(self.test_config['save_result_path'], exist_ok=True)
        self.device = torch.device('cuda:%d' % self.test_config['device_ids'][0] if torch.cuda.is_available else 'cpu')
        self.model = nn.DataParallel(self.model, device_ids=self.test_config['device_ids'])
        self.model.load_state_dict(torch.load(self.test_config['test_pretrain'], map_location=self.device)['model'])
        self.model.to(self.device)
        self.model.eval()
        self.color_map = np.array(self.valid_set.color, dtype=np.uint8)
        self.crop = self.test_config['test_stay_size']
        self.size = self.test_config['test_image_size']

    def process_img(self, img):
        """
        根据测试图像模式处理图像
        img:PIL.Image
        """
        if self.image_config['image_mode'] == 'Gray':
            img = img.convert('L')
            img = Image.merge(mode='RGB', bands=(img, img, img))
        return img

    def getMetric(self):
        """
        根据测试结果和标签文件计算指标
        """
        if os.path.isdir(self.test_config['label_path']):
            self.lab_list = os.listdir(self.test_config['label_path'])
            self.pre_list = os.listdir(self.test_config['save_result_path'])
            self.lab_list.sort()
            self.pre_list.sort()
            assert len(self.lab_list) == len(self.pre_list)
            self.metric.reset()
            for lab_name, pre_name in zip(self.lab_list, self.pre_list):
                lab = np.array(Image.open(os.path.join(self.test_config['label_path'], lab_name)).convert('L'))
                # print(np.unique(lab))
                # lab = self.valid_set.change_label(lab)
                pre = np.array(Image.open(os.path.join(self.test_config['save_result_path'], pre_name)).convert('L'))
                # print(np.unique(pre))
                pre = self.valid_set.label2index(pre)
                self.metric.add(pre, lab)
            miou, ious = self.metric.miou()
            fw_iou = self.metric.fw_iou()
            pa = self.metric.pixel_accuracy()
            mpa = self.metric.pixel_accuracy_class()
            ious = [round(x, 4) for x in ious]
            print("pa=%.4f, mpa=%.4f, fw_iou=%.4f, miou=%.4f, ious=%s" % (pa, mpa, fw_iou, miou, ious))
        else:
            lab = np.array(Image.open(self.test_config['label_path']).convert('L'))
            lab = self.valid_set.change_label(lab)
            pre = np.array(Image.open(self.test_config['save_result_path']).convert('L'))
            pre = self.valid_set.change_label(pre)
            self.metric.reset()
            self.metric.add(pre, lab)
            miou, ious = self.metric.miou()
            fw_iou = self.metric.fw_iou()
            pa = self.metric.pixel_accuracy()
            mpa = self.metric.pixel_accuracy_class()
            ious = [round(x, 4) for x in ious]
            print("pa=%.4f, mpa=%.4f, fw_iou=%.4f, miou=%.4f, ious=%s" % (pa, mpa, fw_iou, miou, ious))

    def __call__(self, need_crop=True):
        if need_crop:  # 测试图像是否需要裁剪
            if os.path.isdir(self.test_config['test_path']):
                # 测试一批图像
                img_list = os.listdir(self.test_config['test_path'])
                for img_name in tqdm(img_list):
                    img_path = os.path.join(self.test_config['test_path'], img_name)
                    img = Image.open(img_path)
                    img = self.process_img(img)
                    img = np.array(img)
                    self.testBig(img)

                    color_map = self.color_map[self.mask]
                    save_name = img_name.split('.')[0] + self.test_config['result_suffix']
                    Image.fromarray(color_map).save(os.path.join(self.test_config['save_result_path'], save_name))
            else:  # 测试一副图像, 默认结果存在同一文件夹下
                img = Image.open(self.test_config['test_path'])
                img = self.process_img(img)
                img = np.array(img)
                self.testBig(img)

                color_map = self.color_map[self.mask]
                base_name = os.path.basename(self.test_config['test_path'])
                base_path = self.test_config['test_path'].rstrip(base_name)
                save_name = base_name.split('.')[0] + self.test_config['result_suffix']
                Image.fromarray(color_map).save(os.path.join(base_path, save_name))
        else:
            self.test_loader = DataLoader(LoadTestData(self.test_config['test_path']),
                                          batch_size=self.test_config['batch_size'],
                                          num_workers=4 * len(self.test_config['device_ids'])
                                          )
            self.get_batch()

    def testBig(self, img):
        '''采用滑窗的方式测试大图'''
        shape = img.shape
        pad0 = int(np.ceil(shape[0] / self.crop) * self.crop - shape[0])
        pad1 = int(np.ceil(shape[1] / self.crop) * self.crop - shape[1])

        pad = int((self.test_config['test_image_size'] - self.crop) / 2)
        padding = ((pad, pad0 + pad), (pad, pad1 + pad), (0, 0))

        new_img = np.pad(img, padding, mode='reflect')
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
                patch = self.train_set.normal(patch)
                patch = torch.from_numpy(patch).float().unsqueeze(dim=0).permute((0, 3, 1, 2))
                bs_patch.append(patch)
                cnt += 1
                self.index.append((i, j))
                if cnt == self.test_config['batch_size']:
                    bs_patch_tensor = torch.cat(bs_patch, dim=0)
                    patch_pre = self.testBatch(bs_patch_tensor)
                    patch_pre = patch_pre[:, pad: pad + self.crop, pad: pad + self.crop]  # 保留中间部分，去除边缘
                    self.puzzle(patch_pre)
                    self.index.clear()
                    bs_patch.clear()
                    cnt = 0
        if len(self.index) > 0:
            bs_patch_tensor = torch.cat(bs_patch, dim=0)
            patch_pre = self.testBatch(bs_patch_tensor)
            try:
                patch_pre = patch_pre[:, pad: pad + self.crop, pad: pad + self.crop]
            except:
                patch_pre = patch_pre[pad: pad + self.crop, pad: pad + self.crop]
            self.puzzle(patch_pre)
        self.mask = self.mask[:shape[0], :shape[1]]

    def puzzle(self, pre):
        """将测试结果拼接回去"""
        for k, (i, j) in enumerate(self.index):
            try:
                self.mask[i: i + self.crop, j: j + self.crop] += pre[k, :, :]
            except:
                self.mask[i: i + self.crop, j: j + self.crop] += pre[:, :]

    def testBatch(self, img):
        """测试一个batch的图像"""
        with torch.no_grad():
            img = img.to(self.device)
            final_out = torch.zeros(size=(img.size(0), self.model_config['num_classes'], img.size(2), img.size(3)))
            """多尺度测试"""
            for scale in self.test_config['multi_scale']:
                new_size = int(img.size(2) * scale)
                new_batch = self.resize_tensor(img, new_size)
                new_out = self.model(new_batch)
                if isinstance(new_out, tuple):
                    prob = new_out[1]
                else:
                    prob = new_out
                new_out = self.resize_tensor(prob, img.size(2))
                final_out += new_out.cpu()
            _, out = torch.max(final_out, dim=1)
            pre_array = out.squeeze().cpu().numpy().astype(np.uint8)
            return pre_array

    def resize_tensor(self, tensor, size):
        new_tensor = F.interpolate(tensor, size=size, mode='bilinear', align_corners=True)
        return new_tensor

    def get_batch(self):
        """测试小图"""
        for img, name in tqdm(self.test_loader):
            batch_pre = self.testBatch(img)
            self.save_res(batch_pre, name)

    def save_res(self, pre, name_list):
        """按照相同的名字保存测试结果"""
        for i, name in enumerate(name_list):
            mask = pre[i].astype(np.uint8)
            mask = self.color_map[mask]
            save_name = name.split('.')[0] + self.test_config['result_suffix']
            Image.fromarray(mask).save(os.path.join(self.test_config['save_result_path'], save_name))


if __name__ == '__main__':
    infer = Infer(config_path='config/train.yaml')
    # infer = Infer(config_path='config/hrnetv2_edge.yaml')
    # infer(need_crop=False)
    infer.getMetric()
