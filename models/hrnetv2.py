from encoder.hrnet.hrnet import get_seg_model
import torch


def HRnetv2(in_channels, num_classes, use_ocr_head):
    net = get_seg_model(in_feats=in_channels, num_classes=num_classes, use_ocr_head=use_ocr_head)
    return net


if __name__ == '__main__':
    hrnet = HRnetv2(3, 10, False)
    x = torch.rand([1, 3, 256, 256])
    x = hrnet(x)
    print(x.size())