import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.ocr_loss.rmi import RMILoss


class SegmentationLosses(object):
    def __init__(self, weight=None, batch_average=True, ignore_index=255, cuda=False, device=None, num_classes=10):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda
        self.device = device
        self.RMILoss = RMILoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ImgBasedCE = ImageBasedCrossEntropyLoss2d(classes=num_classes, ignore_index=ignore_index)

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'rmi':
            return self.RMILoss
        elif mode == 'ibce':
            return self.ImgBasedCE
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.to(device=self.device)

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.to(device=self.device)

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def DiceLoss(self, logit, target):
        dice_loss = dice_bce_loss()
        return dice_loss(logit, target)


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_pred, y_true):
        a = 0.2 * self.bce_loss(y_pred, y_true)
        b = 0.8 * self.soft_dice_loss(y_true, y_pred)
        return a + b


class ImageBasedCrossEntropyLoss2d(nn.Module):
    def __init__(self, classes, weight=None, ignore_index=255, norm=False, upper_bound=1.0, fp16=False):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, reduction='mean', ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = True
        self.fp16 = fp16  # 控制精度是否变为16

    def calculate_weight(self, target):
        """
        calculate weights of classes based on the training crop
        """
        bins = torch.histc(target, bins=self.num_classes, min=0.0, max=self.num_classes)
        hist_norm = bins.float() / bins.sum()
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound * (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound * (1. - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets):
        weights = self.calculate_weight(targets)
        loss = 0.0
        for i in range(inputs.shape[0]):
            if self.fp16:
                weights = weights.half()
            self.nll_loss.weight = weights
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1), targets[i].unsqueeze(0),)
        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
