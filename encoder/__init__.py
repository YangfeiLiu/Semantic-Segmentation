from encoder import resnet, xception

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'seresnet50':
        return resnet.SEResNet50(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError