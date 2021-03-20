import torch.nn as nn
from .ocr_utils import SpatialGatherModule, SpatialOCRModule
from models.hrnetv2 import HRNetv2_ORG


def scale_as(x, y):
    y_size = y.size(2), y.size(3)
    x_scaled = nn.functional.interpolate(x, size=y_size, mode='bilinear', align_corners=True)
    return x_scaled


class OCR_block(nn.Module):
    def __init__(self, high_level_ch, num_classes):
        super(OCR_block, self).__init__()

        ocr_mid_channels = 512
        ocr_key_channels = 256
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True)
        )
        self.ocr_gather_head = SpatialGatherModule(num_classes)
        self.ocr_distri_head = SpatialOCRModule(ocr_mid_channels, ocr_key_channels, ocr_mid_channels, scale=1, dropout=0.05)
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_level_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_level_ch, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, high_level_feats):  # 720 64 64
        feats = self.conv3x3_ocr(high_level_feats)  # 512 64 64
        aux_out = self.aux_head(high_level_feats)  # classes 64 64
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class OCRNet(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(OCRNet, self).__init__()
        self.backbone = HRNetv2_ORG(in_feats, num_classes)
        hign_level_ch = self.backbone.last_inp_channels
        self.ocr = OCR_block(hign_level_ch, num_classes=num_classes)

    def forward(self, x):
        high_level_feats, _ = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_feats)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)
        return aux_out, cls_out


if __name__ == '__main__':
    model = OCRNet(1, 11)
    import torch
    x = torch.randn(size=(1, 1, 256, 256))
    y, yy = model(x)
    print(y.size(), yy.size())

