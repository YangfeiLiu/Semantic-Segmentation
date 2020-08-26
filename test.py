import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import os
from models.deeplab import DeepLab
from models.lednet import LEDNet
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from albumentations import Normalize
#
#
# class LoadTestData(Dataset):
#     def __init__(self, ):
#         super(LoadTestData, self).__init__()


class Test():
    def __init__(self, args):
        backbone = args.arch
        self.num_classes = args.num_classes
        self.aug_test = args.aug_test  # 是否在测试时增强数据
        self.pre_save_dir = args.root
        self.size = args.size
        self.crop = args.crop
        os.makedirs(self.pre_save_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        if args.modelname == 'deeplab':
            self.model = DeepLab(backbone=backbone, output_stride=16, num_classes=self.num_classes, freeze_bn=False)
        if args.modelname == 'lednet':
            self.model = LEDNet(num_classes=self.num_classes)
        self.model = nn.DataParallel(self.model, device_ids=args.device_ids).to(self.device)
        self.model.load_state_dict(torch.load(args.checkpoint)['model'])
        self.color_map = np.array([[0,0,0], [0,250,0], [0,150,0], [250,250,250], 
                                   [100,100,100], [200,200,100], [250,0,0], [0,0,250]], dtype=np.uint8)

    def __call__(self, img_path):
        save_path = os.path.dirname(img_path)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        print(image_name)
        img = np.array(Image.open(img_path))
        mask = self.testBig(img)
        color_mask = self.color_map[mask]
        Image.fromarray(color_mask).save(os.path.join(save_path, image_name + '.png'))
        Image.fromarray(mask).save(os.path.join(save_path, image_name + '_mask.png'))

    def normal(self, img):
        mean = (93.35397203, 111.22929651, 92.32306876)
        std  = (24.17318143, 20.92185836, 18.49409081)
        img = Normalize(mean=mean, std=std)(image=img)['image']
        return img

    def testBig(self, img):
        shape = img.shape
        pad0 = int(np.ceil(shape[0] / self.crop) * self.crop - shape[0])
        pad1 = int(np.ceil(shape[1] / self.crop) * self.crop - shape[1])

        pad = int((self.size - self.crop) / 2)
        padding = ((pad, pad0 + pad), (pad, pad1 + pad), (0, 0))

        new_img = np.pad(img, padding, mode='constant')
        mask = np.zeros(shape=(shape[0] + pad0, shape[1] + pad1), dtype=np.uint8)
        for i in range(0, new_img.shape[0], self.crop):
            if i > new_img.shape[0] - self.size:
                break
            for j in range(0, new_img.shape[1], self.crop):
                if j > new_img.shape[1] - self.size:
                    break
                patch = new_img[i: i + self.size, j: j + self.size, :]
                patch_pre = self.testPatch(patch)
                mask[i: i + self.crop, j: j + self.crop] += patch_pre[pad: pad + self.crop, pad: pad + self.crop]
        mask = mask[:shape[0], :shape[1]]
        return mask

    def testPatch(self, img):
        self.model.eval()
        with torch.no_grad():
            img = self.normal(img)
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(dim=0).permute((0, 3, 1, 2))
            img = img.cuda()
            if self.aug_test:
                scales = args.scales
                biases = args.biases
                flip   = args.flip
            else:
                scales = [1.0]
                flip   = False
                biases = [0.0]
            assert len(scales) == len(biases)

            n, c, h, w = img.size()
            probs = []
            for scale, bias in zip(scales, biases):
                new_h, new_w = int(h * scale + bias), int(w * scale + bias)
                new_img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=True)
                prob = self.model(new_img).softmax(dim=1)
                probs.append(prob)

                if flip:
                    flip_img = new_img.flip(3)
                    flip_prob = self.model(flip_img).softmax(dim=1)
                    prob = flip_prob.flip(3)
                    probs.append(prob)
            prob = torch.stack(probs, dim=0).mean(dim=0)

            _, out = torch.max(prob, dim=1)
            pre_array = out.squeeze().cpu().numpy().astype(np.uint8)
            return pre_array


if __name__ == '__main__':
    scales = [0.5, 1.0, 1.5]
    biases = [0] * 3
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--aug_test', default=False)
    parser.add_argument('--scales', default=scales)
    parser.add_argument('--biases', default=biases)
    parser.add_argument('--flip', default=True)
    parser.add_argument('--size', default=512)
    parser.add_argument('--crop', default=100)
    parser.add_argument('--arch', default='resnet50', choices=['resnet101', 'resnet50', 'seresnet50'],
                       help='deeplab backbone')
    parser.add_argument('--modelname', default='lednet', choices=['deeplab', 'lednet'])
    parser.add_argument('--root', default='/media/hp/1500/liuyangfei/tianzhidata/变化检测/训练样本数据/01_变化检测/0_样本影像/',
                        help='train data path')
    parser.add_argument('--checkpoint', default='/media/hp/1500/liuyangfei/tianzhidata/地表分类/训练样本数据/model/99.pth',
                        help='path to save checkpoint')
    parser.add_argument('--pre_save_dir', default='/media/hp/1500/liuyangfei/tianzhidata/变化检测/训练样本数据/01_变化检测/0_样本影像/')
    parser.add_argument('--device_ids', default=[0, 1])
    args = parser.parse_args()
    year = ['2016', '2019']
    clip = ['clip_1.tif', 'clip_2.tif']
    
    test = Test(args)
    for y in year:
        for c in clip:
            img_path = os.path.join(args.root, y, y + '_' + c)
            test(img_path)

