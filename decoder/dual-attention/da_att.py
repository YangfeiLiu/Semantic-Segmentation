import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import functional as F


class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.in_channels = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, w*h).permute(0, 2, 1)  # b, w*h, c
        proj_key = self.key_conv(x).view(b, -1, w*h)  # b, c, w*h
        energy = torch.bmm(proj_query, proj_key)  # b, w*h, w*h
        attention = self.softmax(energy)  # b, w*h, w*h
        proj_value = self.value_conv(x).view(b, -1, w*h)  # b, c, w*h

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # b, c, w*h
        out = out.view(b, c, h, w)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.in_channels = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = x.view(b, c, -1)  # b, c, h*w
        proj_key = x.view(b, c, -1).permute(0, 2, 1)  # b, h*w, c
        energy = torch.bmm(proj_query, proj_key).permute(0, 2, 1)  # b, c, c
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)  # b, c, c
        proj_value = x.view(b, c, -1)  # b, c, h*w

        out = torch.bmm(attention, proj_value)  # b, c, h, w
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out

