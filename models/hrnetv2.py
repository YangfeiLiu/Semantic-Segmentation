from encoder.hrnet.hrnet import get_seg_model
import torch

def HRnetv2(**kwargs):
    net = get_seg_model(**kwargs)
    return net

if __name__ == '__main__':
    hrnet = HRnetv2()
    x = torch.rand([1, 3, 375, 375])
    x = hrnet(x)
    print(x.size())