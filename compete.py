from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from load_data import CompeteData
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
import sys


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(7)
torch.cuda.manual_seed(7)


WEIGHT = [54.95459747, 8.59527206, 106.53210831, 246.61238861, 28.59648132, 103.38300323, 6.23130417, 100.6644249]

CompeteMap = {"other": 0, "water": 1, "traffic": 2, "building": 3, "farm": 4, 
              "meadow": 5, "forest": 6, "BareSoil": 7}

class Trainer():
    def __init__(self, args):
        backbone = args.arch
        self.model_path = args.model_path
        os.makedirs(self.model_path, exist_ok=True)
        self.num_classes = args.num_classes
        self.metric = MetricMeter(self.num_classes)
        self.start_epoch = args.start_epoch
        self.epoch = args.epoch
        self.lr = args.lr
        self.best_miou = args.best_miou
        self.pretrain = args.pretrain
        self.threshold = args.threshold
        train_set = CompeteData(root=args.root, phase='train')
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)
        valid_set = CompeteData(root=args.root, phase='valid')
        self.valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)
        if args.modelname == 'deeplab':
            self.model = DeepLab(backbone=backbone, output_stride=16, num_classes=self.num_classes, freeze_bn=False)
            train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]
        if args.modelname == 'lednet':
            self.model = LEDNet(num_classes=args.num_classes)
            train_params = self.model.parameters()
        if args.modelname == 'hrnetv2':
            self.model = HRnetv2()
            train_params = self.model.parameters()
        self.optimizer = Adam(train_params, lr=args.lr, weight_decay=0.00004)
        if args.use_balanced_weights:
            weight = 1 / (np.log(np.array(WEIGHT)) + 1.02)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        # self.criterion = SegmentationLosses(weight=weight, cuda=True).build_loss(mode=args.loss_type)
        self.criterion = dice_bce_loss()
        self.model = nn.DataParallel(self.model, device_ids=args.device_ids).to(self.device)
        
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
        # train_accu = 0.0
        # train_miou = 0.0
        tbar = tqdm(self.train_loader)
        tbar.set_description('Training')
        self.model.train()
        for i, (image, mask) in enumerate(tbar):
            image = image.to(self.device)
            mask  = mask.to(self.device)
            self.optimizer.zero_grad()
            out = self.model.forward(image).squeeze()
            loss = self.criterion(mask, out)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                train_loss = ((train_loss * i) + loss.item()) / (i + 1)
                _, out = torch.max(out, dim=1)
                # out[out >= self.threshold] = 1
                # out[out <  self.threshold] = 0
                # train_accu = ((train_accu * i) + accuracy_check_for_batch(mask, out)) / (i + 1)
                # train_miou = ((train_miou * i) + IoU(out, mask, 2)[1]) / (i + 1)
                self.metric.add(out.cpu().numpy(), mask.cpu().numpy())
                train_miou, train_ious = self.metric.miou()  
        # print('train loss=%.6f\t train accu=%.6f\t train miou=%.6f' % (train_loss, train_accu, train_miou))              
        print('Train loss: %.4f' % train_loss, end='\n')
        print('Train miou: %.4f' % train_miou, end='\n')
        print('Train ious: ', train_ious)
        print(CompeteMap.keys())
        print(train_ious[CompeteMap.values()])
    
    def valid_epoch(self):
        self.metric.reset()
        valid_loss = 0.0
        # valid_accu = 0.0
        # valid_miou = 0.0
        tbar = tqdm(self.valid_loader)
        tbar.set_description('Validing')
        batches = len(self.valid_loader)
        self.model.eval()
        with torch.no_grad():
            for i, (image, mask) in enumerate(tbar):
                image = image.to(self.device)
                mask  = mask.to(self.device)
                out = self.model.forward(image).squeeze()
                loss = self.criterion(mask, out)

                valid_loss = ((valid_loss * i) + loss.data) / (i + 1)
                _, out = torch.max(out, dim=1)
                # out[out >= self.threshold] = 1
                # out[out <  self.threshold] = 0
                # valid_accu = ((valid_accu * i) + accuracy_check_for_batch(mask, out)) / (i + 1)
                # valid_miou = ((valid_miou * i) + IoU(out, mask, 2)[1]) / (i + 1)
                self.metric.add(out.cpu().numpy(), mask.cpu().numpy())
                valid_miou, valid_ious = self.metric.miou() 

            print('valid loss: %.4f' % valid_loss, end='\n')
            print('valid miou: %.4f' % valid_miou, end='\n')
            print('valid ious: %s' % valid_ious)
            # print('valid loss=%.6f\t valid accu=%.6f\t valid miou=%.6f' % (valid_loss, valid_accu, valid_miou)) 
        return valid_miou
    
    def save_checkpoint(self, epoch, best_miou):
        meta = {'epoch': epoch,
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'bmiou': best_miou}
        torch.save(meta, os.path.join(self.model_path, '%d.pth' % epoch))
    
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
        milestone = 15  #after epoch milestone, lr is reduced exponentially
        if epoch > milestone:
            lr = self.lr * (0.95 ** (epoch-milestone))
            wd = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = wd

