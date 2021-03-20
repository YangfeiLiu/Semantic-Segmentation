import torch.nn as nn


class DUC(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super(DUC, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pixel_shuffle(x)
        return x
