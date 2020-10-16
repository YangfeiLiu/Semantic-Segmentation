from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from load_data import CompeteData, LoadTestData, MyData
import torch
import torch.nn as nn
from loss import SegmentationLosses, dice_bce_loss
import metrics
from tqdm import tqdm
import numpy as np
import os
from metrics import MetricMeter, accuracy_check_for_batch, IoU
from models.deeplab import DeepLab
from models.lednet import LEDNet
from models.hrnetv2 import HRnetv2
from models.ocrnet import get_seg_model
from models.dinknet import get_dink_model
from ocr_loss.rmi import RMILoss
import sys
from PIL import Image
import cv2

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
torch.manual_seed(7)
torch.cuda.manual_seed(7)

WEIGHT = [8., 8., 8., 1., 8., 8., 8., 8., 8., 8., 8.]

CompeteMap = {"other": 0, "building": 1, "water": 2, "cover": 3, "meadow": 4,
              "soil": 5, "high-road": 6, "road": 7, "handi": 8, "water-farm": 9,
              "shidi": 10}


class Trainer():
    def __init__(self, args):
        arch = args.arch
        backbone = args.backbone
        self.args = args
        os.makedirs(args.model_path, exist_ok=True)
        self.metric = MetricMeter(args.num_classes)
        self.start_epoch = args.start_epoch
        self.epoch = args.epoch
        self.lr = args.lr
        self.best_miou = args.best_miou
        self.pretrain = args.pretrain
        self.threshold = args.threshold
        train_set = MyData(root=args.root, phase='train')
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
        valid_set = MyData(root=args.root, phase='valid')
        self.valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
        if args.modelname == 'deeplab':
            self.model = DeepLab(in_channels=args.in_channels, backbone=arch, output_stride=args.output_stride, num_classes=args.num_classes)
        if args.modelname == 'lednet':
            self.model = LEDNet(num_classes=args.num_classes)
        if args.modelname == 'hrnetv2':
            self.model = HRnetv2(in_channels=args.in_channels, num_classes=args.num_classes, use_ocr_head=False)
        if args.modelname == 'ocrnet':
            self.model = get_seg_model(in_channels=args.in_channels, num_classes=args.num_classes, use_ocr_head=True)
        if args.modelname == 'dinknet':
            self.model = get_dink_model(in_channels=args.in_channels, num_classes=args.num_classes, backbone=backbone)
        train_params = self.model.parameters()
        self.optimizer = Adam(train_params, lr=args.lr, weight_decay=0.0004)
        if args.use_balanced_weights:
            weight = 1 / (np.log(np.array(WEIGHT)) + 1.02)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.criterion = SegmentationLosses(weight=weight, cuda=True).build_loss(mode=args.loss_type)
        self.ocr_criterion = RMILoss(num_classes=args.num_classes).cuda()
        if len(args.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=args.device_ids).to(self.device)
        else:
            self.model.to(self.device)

    def __call__(self):
        cnt = 0
        if self.pretrain is not None:
            print("loading pretrain %s" % self.pretrain)
            self.load_checkpoint(use_optimizer=True, use_epoch=True, use_miou=True)
            print("loaded pretrain %s" % self.pretrain)
        for epoch in range(self.start_epoch, self.epoch):
            print('epoch=%d\t lr=%.6f\t cnt=%d' % (epoch, self.optimizer.param_groups[0]['lr'], cnt))
            self.adjust_learning_rate(self.optimizer, epoch)
            self.train_epoch()
            valid_miou = self.valid_epoch()
            if valid_miou > self.best_miou:
                cnt = 0
                self.save_checkpoint(epoch, valid_miou)
                print('%d.pth saved' % epoch)
                self.best_miou = valid_miou
            else:
                cnt += 1
                if cnt == 20:
                    print('early stop')
                    break

    def train_epoch(self):
        self.metric.reset()
        train_loss = 0.0
        train_miou = 0.0
        tbar = tqdm(self.train_loader)
        self.model.train()
        for i, (image, mask) in enumerate(tbar):
            tbar.set_description('TrainMiou:%.6f' % train_miou)
            image = image.to(self.device)
            mask = mask.to(self.device)
            self.optimizer.zero_grad()
            if self.args.modelname == 'ocrnet':
                aux_out, out = self.model(image)
                aux_loss = self.criterion(aux_out, mask)
                cls_loss = self.criterion(out, mask)
                loss = 0.4 * aux_loss + cls_loss
                loss = loss.mean()
            else:
                out = self.model(image)
                loss = self.criterion(out, mask)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                train_loss = ((train_loss * i) + loss.item()) / (i + 1)
                if self.args.modelname == 'dinknet':
                    out[out >= self.threshold] = 1
                    out[out <  self.threshold] = 0
                else:
                    _, out = torch.max(out, dim=1)
                self.metric.add(out.cpu().numpy(), mask.cpu().numpy())
                train_miou, train_ious = self.metric.miou()
                train_fwiou = self.metric.fw_iou()
        print('Train loss: %.4f' % train_loss, end='\t')
        print('Train FWiou: %.4f' % train_fwiou, end='\t')
        print('Train miou: %.4f' % train_miou, end='\n')
        for cls in CompeteMap.keys():
            print('%10s' % cls + '\t' + '%.6f' % train_ious[CompeteMap[cls]])

    def valid_epoch(self):
        self.metric.reset()
        valid_loss = 0.0
        valid_miou = 0.0
        tbar = tqdm(self.valid_loader)
        self.model.eval()
        with torch.no_grad():
            for i, (image, mask) in enumerate(tbar):
                tbar.set_description('ValidMiou:%.6f' % valid_miou)
                image = image.to(self.device)
                mask = mask.to(self.device)
                if self.args.modelname == 'ocrnet':
                    aux_out, out = self.model(image)
                    aux_loss = self.criterion(aux_out, mask)
                    cls_loss = self.criterion(out, mask)
                    loss = 0.4 * aux_loss + cls_loss
                    loss = loss.mean()
                else:
                    out = self.model(image)
                    loss = self.criterion(out, mask)
                valid_loss = ((valid_loss * i) + loss.data) / (i + 1)
                if self.args.modelname == 'dinknet':
                    out[out >= self.threshold] = 1
                    out[out <  self.threshold] = 0
                else:
                    _, out = torch.max(out, dim=1)
                self.metric.add(out.cpu().numpy(), mask.cpu().numpy())
                valid_miou, valid_ious = self.metric.miou()
                valid_fwiou = self.metric.fw_iou()

            print('valid loss: %.4f' % valid_loss, end='\t')
            print('valid fwiou: %.4f' % valid_fwiou, end='\t')
            print('valid miou: %.4f' % valid_miou, end='\n')
            for cls in CompeteMap.keys():
                print('%10s' % cls + '\t' + '%.6f' % valid_ious[CompeteMap[cls]])
        return valid_fwiou

    def save_checkpoint(self, epoch, best_miou):
        meta = {'epoch': epoch,
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'bmiou': best_miou}
        torch.save(meta, os.path.join(self.args.model_path, '%d.pth' % epoch))

    def load_checkpoint(self, use_optimizer, use_epoch, use_miou):
        state_dict = torch.load(self.pretrain)
        self.model.load_state_dict(state_dict['model'])
        if use_optimizer:
            self.optimizer.load_state_dict(state_dict['optim'])
        if use_epoch:
            self.start_epoch = state_dict['epoch'] + 1
        if use_miou:
            self.best_miou = state_dict['bmiou']

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr
        wd = 1e-4
        milestone = 15  # after epoch milestone, lr is reduced exponentially
        if epoch > milestone:
            lr = self.lr * (0.98 ** (epoch - milestone))
            wd = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = wd
