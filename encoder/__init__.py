from encoder import resnet, xception


def build_backbone(in_feats, backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(in_feats, output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(in_feats, output_stride, BatchNorm, pretrained=False)
    elif backbone == 'seresnet50':
        return resnet.SEResNet50(in_feats, output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(in_feats, output_stride, BatchNorm)
    else:
        raise NotImplementedError