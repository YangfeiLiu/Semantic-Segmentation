from decoder.ocr.ocrnet import OCRNet
from utils.modelProperty import count_params


def get_ocr_model(in_channels, num_classes):
    return OCRNet(in_feats=in_channels, num_classes=num_classes)


if __name__ == '__main__':
    model = get_ocr_model(3, 11)
    count_params(model, input_size=512)
    # import torch
    # x = torch.randn(size=(1, 1, 256, 256))
    # y, yy = model(x)
    # print(y.size(), yy.size())
