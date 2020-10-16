from decoder.ocr.ocrnet import OCRNet


def get_seg_model(in_channels, num_classes, use_ocr_head):
    return OCRNet(in_feats=in_channels, num_classes=num_classes, use_ocr_head=use_ocr_head)


if __name__ == '__main__':
    model = get_seg_model(1, 11)
    import torch
    x = torch.randn(size=(1, 1, 256, 256))
    y, yy = model(x)
    print(y.size(), yy.size())
