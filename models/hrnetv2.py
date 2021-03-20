from encoder.hrnet.hrnet import get_hrnetv2_backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modelProperty import count_params


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def trainable(module):
    for p in module.parameters():
        p.requires_grad = False


class HRNetv2_ORG(nn.Module):
    '''最原始的HRNet，输出是输入的1/4'''
    def __init__(self, in_channels, num_classes, cfg_name=''):
        super(HRNetv2_ORG, self).__init__()
        if cfg_name == '':
            self.hrnet_backbone = get_hrnetv2_backbone(in_feats=in_channels)
        else:
            self.hrnet_backbone = get_hrnetv2_backbone(in_feats=in_channels, cfg_name=cfg_name)
        self.last_inp_channels = self.hrnet_backbone.last_inp_channels

        self.cat_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.last_inp_channels,
                out_channels=self.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(self.last_inp_channels),
            nn.ReLU(inplace=True),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                    in_channels=self.last_inp_channels,
                    out_channels=self.last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            nn.BatchNorm2d(self.last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                    in_channels=self.last_inp_channels,
                    out_channels=num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            )
        init_weights(self)
        # trainable(self.hrnet_backbone)

    def forward(self, x):
        x = self.hrnet_backbone(x)  # x is a list which contains 4 outputs with different size
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        cat_feats = torch.cat([x[0], x1, x2, x3], 1)

        cat_feats = self.cat_conv(cat_feats)
        out = self.last_layer(cat_feats)
        return cat_feats, out


class HRNetv2_UP(nn.Module):
    '''这个模块在HRNet后接了一个4倍上采样'''
    def __init__(self, in_channels, num_classes, cfg_name=''):
        super(HRNetv2_UP, self).__init__()
        if cfg_name == '':
            self.backbone = HRNetv2_ORG(in_channels, num_classes)
        else:
            self.backbone = HRNetv2_ORG(in_channels, num_classes, cfg_name)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        cat_feats, _ = self.backbone(x)
        cat_feats_up = F.interpolate(cat_feats, size=(h, w), mode='bilinear', align_corners=True)
        out = self.backbone.last_layer(cat_feats_up)
        return cat_feats, out


class MSFE(nn.Module):
    '''这个模块接在HRNet提取的特征之后'''
    def __init__(self, in_channel, out_channel):
        '''stride:HRNet输出步长'''
        super(MSFE, self).__init__()
        self.out_channels = out_channel
        self.dilate1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.dilate2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=5, padding=5),
            nn.BatchNorm2d(out_channel)
        )
        self.dilate3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=9, padding=9),
            nn.BatchNorm2d(out_channel)
        )
        self.dilate4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=13, padding=13),
            nn.BatchNorm2d(out_channel)
        )
        init_weights(self)

    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        out = F.relu(out, inplace=True)
        return out


class HRNetv2_DUC(nn.Module):
    def __init__(self, in_channels, num_classes, scale_factor=4, cfg_name=''):
        super(HRNetv2_DUC, self).__init__()
        if cfg_name == '':
            self.backbone = HRNetv2_ORG(in_channels, num_classes)
        else:
            self.backbone = HRNetv2_ORG(in_channels, num_classes, cfg_name)
        out_channels = self.backbone.last_inp_channels
        self.msfe = MSFE(out_channels, out_channels // 2)
        self.num_classes = num_classes
        self.scale_factor = scale_factor

        out_channel = self.msfe.out_channels
        out_channel1 = scale_factor * scale_factor * num_classes
        out_channel2 = scale_factor * scale_factor
        self.conv1 = nn.Conv2d(out_channel, out_channel1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel2, 1)
        self.bn2 = nn.BatchNorm2d(out_channel2)
        self.sigmoid = nn.Sigmoid()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        cat_feats, out = self.backbone(x)
        msfe_out = self.msfe(cat_feats)
        body = self.relu(self.bn1(self.conv1(msfe_out)))
        edge = self.sigmoid(self.bn2(self.conv2(msfe_out)))
        final_body = F.pixel_shuffle(body, upscale_factor=self.scale_factor)
        final_edge = F.pixel_shuffle(edge, upscale_factor=self.scale_factor)
        return final_edge, final_body


if __name__ == '__main__':
    hrnet = HRNetv2_ORG(3, 6)
    hrnet_duc = HRNetv2_DUC(3, 6)
    hrnet_up = HRNetv2_UP(3, 6)
    x = torch.rand([1, 3, 512, 512])
    x, y = hrnet(x)
    print(x.size(), y.size())
    count_params(hrnet_duc, input_size=512)
