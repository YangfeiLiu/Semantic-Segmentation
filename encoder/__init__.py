from encoder import resnet, xception
from .resnest.resnest import resnest50, resnest101, resnest200, resnest269


def build_backbone(in_feats, backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(in_feats, output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(in_feats, output_stride, BatchNorm, pretrained=False)
    elif backbone == 'seresnet50':
        return resnet.SEResNet50(in_feats, output_stride, BatchNorm)
    elif backbone == 'seresnet101':
        return resnet.SEResNet101(in_feats, output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(in_feats, output_stride, BatchNorm)
    elif backbone == 'resnest50':
        return resnest50(in_feats, output_stride)
    elif backbone == 'resnest101':
        return resnest101(in_feats, output_stride)
    elif backbone == 'resnest200':
        return resnest200(in_feats, output_stride)
    elif backbone == 'resnest269':
        return resnest269(in_feats, output_stride)
    elif backbone == 'resnet50_cbam':
        return resnet.resnet50_cbam(in_feats, output_stride, BatchNorm)
    elif backbone == 'resnet101_cbam':
        return resnet.resnet101_cbam(in_feats, output_stride, BatchNorm)
    else:
        raise NotImplementedError