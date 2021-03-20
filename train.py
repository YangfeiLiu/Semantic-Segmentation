from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import utils
from ReadData import Data
import torch
from loss.loss import SegmentationLosses
from tqdm import tqdm
import numpy as np
from utils.setSeed import setup_seed
from config.load_config import LoadConfig
import torch.nn as nn
import os
from models import getModel
from loguru import logger
from tensorboardX import SummaryWriter
from collections import OrderedDict


setup_seed(20)


class Trainer():
    def __init__(self, config_path):
        self.image_config, self.model_config, self.run_config = LoadConfig(config_path=config_path).train_config()
        self.device = torch.device('cuda:%d' % self.run_config['device_ids'][0] if torch.cuda.is_available else 'cpu')
        self.model = getModel(self.model_config)
        os.makedirs(self.run_config['model_save_path'], exist_ok=True)
        if len(self.run_config['device_ids']) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.run_config['device_ids'])
        self.model.to(device=self.device)
        self.run_config['num_workers'] = self.run_config['num_workers'] * len(self.run_config['device_ids'])
        self.train_set = Data(root=self.image_config['image_path'],
                              phase='train',
                              img_mode=self.image_config['image_mode'],
                              n_classes=self.model_config['num_classes'],
                              size=self.image_config['image_size'],
                              scale=self.image_config['image_scale'])
        self.valid_set = Data(root=self.image_config['image_path'],
                              phase='valid',
                              img_mode=self.image_config['image_mode'],
                              n_classes=self.model_config['num_classes'],
                              size=self.image_config['image_size'],
                              scale=self.image_config['image_scale'])
        self.valid_set(self.image_config['data_name'])
        self.className = self.valid_set.className
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.run_config['batch_size'],
                                       shuffle=True,
                                       num_workers=self.run_config['num_workers'],
                                       pin_memory=True,
                                       drop_last=False)
        self.valid_loader = DataLoader(self.valid_set,
                                       batch_size=self.run_config['batch_size'],
                                       shuffle=True,
                                       num_workers=self.run_config['num_workers'],
                                       pin_memory=True,
                                       drop_last=False)
        train_params = self.model.parameters()
        self.optimizer = Adam(train_params, lr=eval(self.run_config['lr']), weight_decay=self.run_config['weight_decay'])
        self.lr_scheduler = utils.adjustLR.AdjustLr(self.optimizer)
        if self.run_config['use_weight_balance']:
            weight = utils.weight_balance.getWeight(self.run_config['weights_file'])
        else:
            weight = None
        self.Criterion = SegmentationLosses(weight=weight, cuda=True, device=self.device, batch_average=False)
        self.metric = utils.metrics.MetricMeter(self.model_config['num_classes'])

    @logger.catch  # 在日志中记录错误
    def __call__(self):
        self.global_name = self.model_config['model_name']
        logger.add(os.path.join(self.image_config['image_path'], 'log', 'log_' + self.global_name + '/train_{time}.log'),
                   format="{time} {level} {message}", level="INFO", encoding='utf-8')
        self.writer = SummaryWriter(logdir=os.path.join(self.image_config['image_path'], 'run', 'runs_' + self.global_name))
        logger.info("image_config: {} \n model_config: {} \n run_config: {}", self.image_config, self.model_config,
                    self.run_config)
        cnt = 0
        if self.run_config['pretrain'] != '':
            logger.info("loading pretrain %s" % self.run_config['pretrain'])
            try:
                self.load_checkpoint(use_optimizer=False, use_epoch=False, use_miou=False)
            except:
                self.load_checkpoint_with_changed(use_optimizer=False, use_epoch=False, use_miou=False)
        logger.info("start training")
        lr = self.optimizer.param_groups[0]['lr']
        for epoch in range(self.run_config['start_epoch'], self.run_config['epoch']):
            print('epoch=%d, lr=%.8f' % (epoch, lr))
            self.train_epoch(epoch, lr)
            valid_miou = self.valid_epoch(epoch)
            self.lr_scheduler.LambdaLR_(milestone=5, gamma=0.92).step(epoch=epoch)
            self.save_checkpoint(epoch, valid_miou, 'last_' + self.global_name)
            if valid_miou > self.run_config['best_miou']:
                cnt = 0
                self.save_checkpoint(epoch, valid_miou, 'best_' + self.global_name)
                logger.info("#############   %d saved   ##############" % epoch)
                self.run_config['best_miou'] = valid_miou
            else:
                cnt += 1
                if cnt == self.run_config['early_stop']:
                    logger.info("early stop")
                    break
        self.writer.close()

    def train_epoch(self, epoch, lr):
        self.metric.reset()
        train_loss = 0.0
        train_miou = 0.0
        tbar = tqdm(self.train_loader)
        self.model.train()
        for i, (image, mask, edge) in enumerate(tbar):
            tbar.set_description('train_miou:%.6f' % train_miou)
            tbar.set_postfix({"train_loss": train_loss})
            image = image.to(self.device)
            mask = mask.to(self.device)
            edge = edge.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(image)
            if isinstance(out, tuple):
                aux_out, final_out = out[0], out[1]
            else:
                aux_out, final_out = None, out
            if self.model_config['model_name'] == 'ocrnet':
                aux_loss = self.Criterion.build_loss(mode='rmi')(aux_out, mask)
                cls_loss = self.Criterion.build_loss(mode='rmi')(final_out, mask)
                loss = 0.4 * aux_loss + cls_loss
                loss = loss.mean()
            elif self.model_config['model_name'] == 'hrnetv2_duc':
                loss_body = self.Criterion.build_loss(mode=self.run_config['loss_type'])(final_out, mask)
                loss_edge = self.Criterion.build_loss(mode='dice')(aux_out, edge)
                loss = loss_body + loss_edge
            else:
                loss = self.Criterion.build_loss(mode=self.run_config['loss_type'])(final_out, mask)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                train_loss = ((train_loss * i) + loss.item()) / (i + 1)
                if self.model_config['model_name'] == 'dinknet' and self.model_config['num_classes'] == 1:
                    final_out[final_out >= self.run_config['threshold']] = 1
                    final_out[final_out <  self.run_config['threshold']] = 0
                else:
                    _, pred = torch.max(final_out, dim=1)
                self.metric.add(pred.cpu().numpy(), mask.cpu().numpy())
                train_miou, train_ious = self.metric.miou()
                train_fwiou = self.metric.fw_iou()
                train_accu = self.metric.pixel_accuracy()
                train_fwaccu = self.metric.pixel_accuracy_class()
        logger.info("Epoch:%2d\t lr:%.8f\t Train loss:%.4f\t Train FWiou:%.4f\t Train Miou:%.4f\t Train accu:%.4f\t "
                    "Train fwaccu:%.4f" % (epoch, lr, train_loss, train_fwiou, train_miou, train_accu, train_fwaccu))
        cls = ""
        ious = list()
        ious_dict = OrderedDict()
        for i, c in enumerate(self.className):
            ious_dict[c] = train_ious[self.className[i]]
            ious.append(ious_dict[c])
            cls += "%s:" % c + "%.4f "
        ious = tuple(ious)
        logger.info(cls % ious)
        # tensorboard
        self.writer.add_scalar("lr", lr, epoch)
        self.writer.add_scalar("loss/train_loss", train_loss, epoch)
        self.writer.add_scalar("miou/train_miou", train_miou, epoch)
        self.writer.add_scalar("fwiou/train_fwiou", train_fwiou, epoch)
        self.writer.add_scalar("accuracy/train_accu", train_accu, epoch)
        self.writer.add_scalar("fwaccuracy/train_fwaccu", train_fwaccu, epoch)
        self.writer.add_scalars("ious/train_ious", ious_dict, epoch)

    def valid_epoch(self, epoch):
        self.metric.reset()
        valid_loss = 0.0
        valid_miou = 0.0
        tbar = tqdm(self.valid_loader)
        self.model.eval()
        with torch.no_grad():
            for i, (image, mask, edge) in enumerate(tbar):
                tbar.set_description('valid_miou:%.6f' % valid_miou)
                tbar.set_postfix({"valid_loss": valid_loss})
                image = image.to(self.device)
                mask = mask.unsqueeze(1).to(self.device)
                edge = edge.to(self.device)
                out = self.model(image)
                if isinstance(out, tuple):
                    aux_out, final_out = out[0], out[1]
                else:
                    aux_out, final_out = _, out
                if self.model_config['model_name'] == 'ocrnet':
                    aux_loss = self.Criterion.build_loss(mode='rmi')(aux_out, mask)
                    cls_loss = self.Criterion.build_loss(mode='rmi')(final_out, mask)
                    loss = 0.4 * aux_loss + cls_loss
                    loss = loss.mean()
                elif self.model_config['model_name'] == 'hrnetv2_duc':
                    loss_body = self.Criterion.build_loss(mode='ce')(final_out, mask)
                    loss_edge = self.Criterion.build_loss(mode='dice')(aux_out, edge)
                    loss = loss_body + loss_edge
                else:
                    loss = self.Criterion.build_loss(mode='ce')(final_out, mask)
                valid_loss = ((valid_loss * i) + float(loss)) / (i + 1)
                if self.model_config['model_name'] == 'dinknet' and self.model_config['num_classes'] == 1:
                    final_out[final_out >= self.run_config['threshold']] = 1
                    final_out[final_out < self.run_config['threshold']] = 0
                else:
                    _, pred = torch.max(final_out, dim=1)
                self.metric.add(pred.cpu().numpy(), mask.cpu().numpy())
                valid_miou, valid_ious = self.metric.miou()
                valid_fwiou = self.metric.fw_iou()
                valid_accu = self.metric.pixel_accuracy()
                valid_fwaccu = self.metric.pixel_accuracy_class()
            logger.info("epoch:%d\t valid loss:%.4f\t valid fwiou:%.4f\t valid miou:%.4f valid accu:%.4f\t "
                        "valid fwaccu:%.4f\t" % (epoch, valid_loss, valid_fwiou, valid_miou, valid_accu, valid_fwaccu))
            ious = list()
            cls = ""
            ious_dict = OrderedDict()
            for i, c in enumerate(self.className):
                ious_dict[c] = valid_ious[self.className[i]]
                ious.append(ious_dict[c])
                cls += "%s:" % c + "%.4f "
            ious = tuple(ious)
            logger.info(cls % ious)
            self.writer.add_scalar("loss/valid_loss", valid_loss, epoch)
            self.writer.add_scalar("miou/valid_miou", valid_miou, epoch)
            self.writer.add_scalar("fwiou/valid_fwiou", valid_fwiou, epoch)
            self.writer.add_scalar("accuracy/valid_accu", valid_accu, epoch)
            self.writer.add_scalar("fwaccuracy/valid_fwaccu", valid_fwaccu, epoch)
            self.writer.add_scalars("ious/valid_ious", ious_dict, epoch)
        return valid_miou

    def save_checkpoint(self, epoch, best_miou, flag):
        meta = {'epoch': epoch,
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'bmiou': best_miou}
        try:
            torch.save(meta, os.path.join(self.run_config['model_save_path'], '%s.pth' % flag),
                       _use_new_zipfile_serialization=False)
        except:
            torch.save(meta, os.path.join(self.run_config['model_save_path'], '%s.pth' % flag))

    def load_checkpoint(self, use_optimizer, use_epoch, use_miou):
        state_dict = torch.load(self.run_config['pretrain'], map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        if use_optimizer:
            self.optimizer.load_state_dict(state_dict['optim'])
        if use_epoch:
            self.run_config['start_epoch'] = state_dict['epoch'] + 1
        if use_miou:
            self.run_config['best_miou'] = state_dict['bmiou']

    def load_checkpoint_with_changed(self, use_optimizer, use_epoch, use_miou):
        state_dict = torch.load(self.run_config['pretrain'], map_location=self.device)
        pretrain_dict = state_dict['model']
        model_dict = self.model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and 'edge' not in k}
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)
        if use_optimizer:
            self.optimizer.load_state_dict(state_dict['optim'])
        if use_epoch:
            self.run_config['start_epoch'] = state_dict['epoch'] + 1
        if use_miou:
            self.run_config['best_miou'] = state_dict['bmiou']


if __name__ == '__main__':
    cfg_path = 'config/train.yaml'
    train = Trainer(config_path=cfg_path)
    train()

