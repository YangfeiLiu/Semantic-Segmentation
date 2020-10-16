import torch.nn as nn
import torch.nn.functional as F
import torch


class Dblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.dilate2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(out_channel)
        )
        self.dilate3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=5, padding=5),
            nn.BatchNorm2d(out_channel)
        )
        self.dilate4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=7, padding=7),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x))
        dilate2_out = F.relu(self.dilate2(dilate1_out))
        dilate3_out = F.relu(self.dilate3(dilate2_out))
        dilate4_out = F.relu(self.dilate4(dilate3_out))
        out = torch.cat([dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)
        out = self.relu(self.bn1(self.conv1(out)))
        return self.dropout(out)


def build_Dblock(in_channel, out_channel):
    return Dblock(in_channel, out_channel)
