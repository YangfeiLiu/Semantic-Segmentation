from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from utils.adjustLR import AdjustLr
from ReadData import MyData
import torch
import torch.nn as nn
from loss.loss import SegmentationLosses
from tqdm import tqdm
import numpy as np
import os
from utils.metrics import MetricMeter
from models.deeplab import DeepLab
from models.lednet import LEDNet
from models.hrnetv2 import HRnetv2
from models.ocrnet import get_seg_model
from models.dinknet import get_dink_model
from loss.ocr_loss.rmi import RMILoss
from loguru import logger
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
torch.manual_seed(7)
torch.cuda.manual_seed(7)

logger.add('./log_resnest101/train_{time}.log', format="{time} {level} {message}", level="INFO")
writer = SummaryWriter(logdir='./runs_resnest101/')

ClassMap = {"meadow": 5, "building": 0, "farm": 1, "water": 3, "forest": 2, "road": 4, "others": 6}
# ColorMap = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)


class Trainer():
    def __init__(self, args):
        logger.info("start to train %s\t ClassMap %s\t batch_size:%d\t in_channels:%d\t num_classes:%d" %
                    (args.modelname, ClassMap, args.batch_size, args.in_channels, args.num_classes))
        self.arch = args.arch
        self.backbone = args.backbone
        self.args = args
        os.makedirs(args.model_path, exist_ok=True)
        self.metric = MetricMeter(args.num_classes)
        self.start_epoch = args.start_epoch
        self.epoch = args.epoch
        self.lr = args.lr
        self.best_miou = args.best_miou
        self.pretrain = args.pretrain
        self.threshold = args.threshold
        train_set = MyData(root=args.root, phase='train', img_mode=args.img_mode, n_classes=args.num_classes, size=args.size, scale=args.scales)
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
        valid_set = MyData(root=args.root, phase='valid', img_mode=args.img_mode, n_classes=args.num_classes, size=args.size, scale=args.scales)
        self.valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
        if args.modelname == 'deeplab':
            self.model = DeepLab(in_channels=args.in_channels, backbone=self.arch, enhance=args.enhance, output_stride=args.output_stride, num_classes=args.num_classes)
        if args.modelname == 'lednet':
            self.model = LEDNet(num_classes=args.num_classes)
        if args.modelname == 'hrnetv2':
            self.model = HRnetv2(in_channels=args.in_channels, num_classes=args.num_classes, use_ocr_head=False)
        if args.modelname == 'ocrnet':
            self.model = get_seg_model(in_channels=args.in_channels, num_classes=args.num_classes, use_ocr_head=True)
        if args.modelname == 'dinknet':
            self.model = get_dink_model(in_channels=args.in_channels, num_classes=args.num_classes, backbone=self.backbone)
        train_params = self.model.parameters()
        self.optimizer = Adam(train_params, lr=args.lr, weight_decay=0.00004)
        self.lr_scheduler = AdjustLr(self.optimizer)
        if args.use_weight_balance:
            with open(args.root + 'weight.txt', 'r') as f:
                line = f.readline()
                weight = line.split(' ')
                weight = [int(x) for x in weight]
                weight = np.array(weight)
                weight = weight / np.min(weight)
                weight = 1 / (np.log(weight) + 1.02)
                weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.device = torch.device('cuda:%d' % args.device_ids[0] if torch.cuda.is_available else 'cpu')
        self.criterion = SegmentationLosses(weight=weight, cuda=True, device=self.device, batch_average=False).build_loss(mode=args.loss_type)
        self.ocr_criterion = RMILoss(num_classes=args.num_classes).to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=args.device_ids).to(self.device)

    @logger.catch  # 在日志中记录错误
    def __call__(self):
        cnt = 0
        if self.pretrain is not None:
            logger.info("loading pretrain %s" % self.pretrain)
            self.load_checkpoint(use_optimizer=True, use_epoch=True, use_miou=False)
        logger.info("start training")
        for epoch in range(self.start_epoch, self.epoch):
            self.train_epoch(epoch, self.optimizer.param_groups[0]['lr'])
            valid_miou = self.valid_epoch(epoch)
            self.lr_scheduler.LambdaLR_().step(epoch=epoch)
            self.save_checkpoint(epoch, valid_miou, 'last_' + self.args.modelname + self.arch)
            if valid_miou > self.best_miou:
                cnt = 0
                self.save_checkpoint(epoch, valid_miou, 'best_' + self.args.modelname + self.arch)
                logger.info("%d saved" % epoch)
                self.best_miou = valid_miou
            else:
                cnt += 1
                if cnt == 20:
                    logger.info("early stop")
                    break
        writer.close()

    def train_epoch(self, epoch, lr):
        self.metric.reset()
        train_loss = 0.0
        train_miou = 0.0
        tbar = tqdm(self.train_loader)
        self.model.train()
        for i, (image, mask) in enumerate(tbar):
            tbar.set_description('TrainMiou:%.6f' % train_miou)
            tbar.set_postfix({"train_loss": train_loss})
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
                out = self.model(image).squeeze()
                loss = self.criterion(out, mask)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                train_loss = ((train_loss * i) + loss.item()) / (i + 1)
                if self.args.modelname == 'dinknet' and self.args.num_classes == 1:
                    out[out >= self.threshold] = 1
                    out[out <  self.threshold] = 0
                else:
                    _, out = torch.max(out, dim=1)
                self.metric.add(out.cpu().numpy(), mask.cpu().numpy())
                train_miou, train_ious = self.metric.miou()
                train_fwiou = self.metric.fw_iou()
                train_accu = self.metric.pixel_accuracy()
                train_fwaccu = self.metric.pixel_accuracy_class()
        logger.info("Epoch:%2d\t lr:%.8f\t Train loss:%.4f\t Train FWiou:%.4f\t Train Miou:%.4f\t Train accu:%.4f\t "
                    "Train fwaccu:%.4f" % (epoch, lr, train_loss, train_fwiou, train_miou, train_accu, train_fwaccu))
        cls = ""
        ious = list()
        ious_dict = dict()

        for c in ClassMap.keys():
            ious_dict[c] = train_ious[ClassMap[c]]
            ious.append(train_ious[ClassMap[c]])
            cls += "%s:" % c + "%.4f "
        ious = tuple(ious)
        logger.info(cls % ious)
        # tensorboard
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("loss/train_loss", train_loss, epoch)
        writer.add_scalar("miou/train_miou", train_miou, epoch)
        writer.add_scalar("fwiou/train_fwiou", train_fwiou, epoch)
        writer.add_scalar("accuracy/train_accu", train_accu, epoch)
        writer.add_scalar("fwaccuracy/train_fwaccu", train_fwaccu, epoch)
        writer.add_scalars("ious/train_ious", ious_dict, epoch)

    def valid_epoch(self, epoch):
        self.metric.reset()
        valid_loss = 0.0
        valid_miou = 0.0
        tbar = tqdm(self.valid_loader)
        self.model.eval()
        with torch.no_grad():
            for i, (image, mask) in enumerate(tbar):
                tbar.set_description('ValidMiou:%.6f' % valid_miou)
                tbar.set_postfix({"valid_loss": valid_loss})
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
                    loss = self.criterion(out.squeeze(), mask)
                valid_loss = ((valid_loss * i) + loss.item()) / (i + 1)
                if self.args.modelname == 'dinknet' and self.args.num_classes == 1:
                    out[out >= self.threshold] = 1
                    out[out <  self.threshold] = 0
                else:
                    _, out = torch.max(out, dim=1)
                # if i == 0:
                #     val_img = make_grid(image, nrow=4, normalize=True)
                #     val_mask = resnest.from_numpy(ColorMap[mask.cpu().numpy()]).float().permute((0, 3, 1, 2))
                #     val_mask = make_grid(val_mask, nrow=4)
                #     val_pred = resnest.from_numpy(ColorMap[out.cpu().numpy()]).float().permute((0, 3, 1, 2))
                #     val_pred = make_grid(val_pred, nrow=4)
                self.metric.add(out.squeeze().cpu().numpy(), mask.cpu().numpy())
                valid_miou, valid_ious = self.metric.miou()
                valid_fwiou = self.metric.fw_iou()
                valid_accu = self.metric.pixel_accuracy()
                valid_fwaccu = self.metric.pixel_accuracy_class()
            logger.info("epoch:%d\t valid loss:%.4f\t valid fwiou:%.4f\t valid miou:%.4f valid accu:%.4f\t "
                        "valid fwaccu:%.4f\t" % (epoch, valid_loss, valid_fwiou, valid_miou, valid_accu, valid_fwaccu))
            ious = list()
            cls = ""
            ious_dict = dict()
            for c in ClassMap.keys():
                ious_dict[c] = valid_ious[ClassMap[c]]
                ious.append(valid_ious[ClassMap[c]])
                cls += "%s:" % c + "%.4f "
            ious = tuple(ious)
            logger.info(cls % ious)

            # tensorboard
            # writer.add_image("valid/image", val_img, epoch)
            # writer.add_image("vaild/mask", val_mask, epoch)
            # writer.add_image("valid/pred", val_pred, epoch)
            writer.add_scalar("loss/valid_loss", valid_loss, epoch)
            writer.add_scalar("miou/valid_miou", valid_miou, epoch)
            writer.add_scalar("fwiou/valid_fwiou", valid_fwiou, epoch)
            writer.add_scalar("accuracy/valid_accu", valid_accu, epoch)
            writer.add_scalar("fwaccuracy/valid_fwaccu", valid_fwaccu, epoch)
            writer.add_scalars("ious/valid_ious", ious_dict, epoch)
        return valid_miou

    def save_checkpoint(self, epoch, best_miou, flag):
        meta = {'epoch': epoch,
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'bmiou': best_miou}
        try:
            torch.save(meta, os.path.join(self.args.model_path, '%s.pth' % flag), _use_new_zipfile_serialization=False)
        except:
            torch.save(meta, os.path.join(self.args.model_path, '%s.pth' % flag))

    def load_checkpoint(self, use_optimizer, use_epoch, use_miou):
        state_dict = torch.load(self.pretrain)
        self.model.load_state_dict(state_dict['model'])
        if use_optimizer:
            self.optimizer.load_state_dict(state_dict['optim'])
        if use_epoch:
            self.start_epoch = state_dict['epoch'] + 1
        if use_miou:
            self.best_miou = state_dict['bmiou']

