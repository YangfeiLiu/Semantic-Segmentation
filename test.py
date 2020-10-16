from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
import os
from models.deeplab import DeepLab
from models.lednet import LEDNet
from models.hrnetv2 import HRnetv2
from models.ocrnet import get_seg_model
from PIL import Image
from tifffile import imread


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
        img = self.normal(img)
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        return img, name

    def __len__(self):
        return len(self.img_list)


class pTester():
    def __init__(self, modelname='ocrnet'):
        backbone = ''
        self.test_root = '/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/tianzhibei/changedatection/TH0102_P201912029046482_1B_G1.tif'
        self.num_classes = 11
        self.pretrain = '/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/code/model/ocr/30.pth'
        self.save_path = '/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/tianzhibei/changedatection/test.png'
        self.size = 512
        self.crop = 300
        self.batch_size = 24
        if modelname == 'deeplab':
            self.model = DeepLab(backbone=backbone, output_stride=8, num_classes=self.num_classes, freeze_bn=False)
        if modelname == 'lednet':
            self.model = LEDNet(num_classes=self.num_classes)
        if modelname == 'hrnetv2':
            self.model = HRnetv2(num_classes=self.num_classes, use_ocr_head=False)
        if modelname == 'ocrnet':
            self.model = get_seg_model(num_classes=self.num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        # self.model = nn.DataParallel(self.model, device_ids=args.device_ids).to(self.device)
        self.model.load_state_dict(torch.load(self.pretrain)['model'])
        self.model.to(self.device)
        # test_data = LoadTestData(self.test_root)
        # self.test_loader = DataLoader(test_data, batch_size=200, num_workers=32, pin_memory=True, drop_last=False)
        self.color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [128, 255, 0], [255, 255, 128],
                                   [128, 128, 255], [128, 128, 0], [255, 0, 255], [0, 128, 255], [64, 128, 64]],
                                  dtype=np.uint8)

    def __call__(self):
        # self.get_batch()
        # img = np.array(Image.open(self.test_root))
        img = imread(self.test_root)
        mask = self.testBig(img)
        color_map = self.color_map[mask]
        Image.fromarray(color_map).save(self.save_path)

    def normal(self, img):
        img = img / 127.5 - 1.
        return img

    '''采用滑窗的方式测试大图'''
    def testBig(self, img):
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
        for i in tqdm(range(0, new_img.shape[0], self.crop)):
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
            patch_pre = patch_pre[:, pad: pad + self.crop, pad: pad + self.crop]
            self.puzzle(patch_pre)
        mask = self.mask[:shape[0], :shape[1]]
        return mask

    def puzzle(self, pre):
        for k, (i, j) in enumerate(self.index):
            self.mask[i: i + self.crop, j: j + self.crop] += pre[k, :, :]

    def testPatch(self, img):
        self.model.eval()
        with torch.no_grad():
            img = img.cuda()
            _, prob = self.model.forward(img)
            _, out = torch.max(prob, dim=1)
            pre_array = out.squeeze().cpu().numpy().astype(np.uint8)
            return pre_array

    '''测试小图'''
    def get_batch(self):
        for img, name in tqdm(self.test_loader):
            img = img.to(self.device)
            batch_pre = self.test(img)
            self.save_res(batch_pre, name)

    def test(self, batch):
        with torch.no_grad():
            # batch = batch.to(self.device)
            out = self.model.forward(batch)
            _, pre = torch.max(out, dim=1)
            pre = pre.squeeze().cpu().detach().numpy()
            return pre

    def save_res(self, pre, name_list):
        for i, name in enumerate(name_list):
            img = pre[i].astype(np.uint16)
            img = (img + 1) * 100
            img = Image.fromarray(img)
            img.save(os.path.join(self.save_path, name.replace('tif', 'png')))


if __name__ == '__main__':
    test = pTester()
    test()
