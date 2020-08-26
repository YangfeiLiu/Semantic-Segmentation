import numpy as np
import cv2
import torch


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if isinstance(item, str):
            item = np.array(cv2.imread(item))
        elif isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())

def accuracy_check_for_batch(masks, predictions):
    total_acc = 0
    for index in range(masks.shape[0]):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc / masks.shape[0]


def IoU(pred, target, n_classes):
    ious = []
    pred = pred.flatten()
    target = target.flatten()

    # Ignore IoU for background class ("0")
    for cls in range(n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_index = pred == cls
        target_index = target == cls
        intersection = (pred_index[target_index]).sum()
        union = pred_index.sum() + target_index.sum() - intersection
        if union == 0:
            ious.append(float(0))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(union))
    return np.array(ious)


class MetricMeter(object):
    """MetricMeter
    """
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.confusion_matrix = np.zeros((self.nclasses,)*2)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def miou(self):
        IoUs = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + np.finfo(np.float32).eps)
        mIoU = np.nanmean(IoUs)
        return mIoU, IoUs

    def fw_iou(self):
        """fw_iou, frequency weighted iou
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        fwIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
        return fwIoU

    def _generate_matrix(self, pred, gt):
        mask = (gt >= 0) & (gt < self.nclasses)
        label = self.nclasses * gt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.nclasses**2)
        confusion_matrix = count.reshape(self.nclasses, self.nclasses)
        return confusion_matrix

    def add(self, pred, gt):
        assert pred.shape == gt.shape
        self.confusion_matrix += self._generate_matrix(pred, gt)

    def reset(self):
        self.confusion_matrix = np.zeros((self.nclasses,) * 2)
