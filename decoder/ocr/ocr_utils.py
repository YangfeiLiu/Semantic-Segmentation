import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGatherModule(nn.Module):
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        bs, c, _, _ = probs.size()

        probs = probs.view(bs, c, -1)
        feats = feats.view(bs, feats.size(1), -1)

        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(self.key_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, proxy):
        bs, c, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(bs, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(bs, self.key_channels, -1)
        value = self.f_down(proxy).view(bs, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(bs, self.key_channels, h, w)
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class SpatialOCRModule(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCRModule, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale)
        _in_channels = in_channels * 2
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        out = self.conv_bn_dropout(torch.cat([context, feats], dim=1))
        return out



