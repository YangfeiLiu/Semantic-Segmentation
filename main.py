import argparse
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation master")
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--img_mode', type=str, default='RGB', choices=["RGB", "Gray"])
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--scales', default=[0.5, 2])
    parser.add_argument('--num_workers', default=8)
    parser.add_argument('--size', default=512)
    parser.add_argument('--use_weight_balance', default=True)
    parser.add_argument('--pretrain', default=None)
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--best_miou', default=0.0)
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--output_stride', default=16, help='deeplab os')
    parser.add_argument('--arch', default='resnest101', choices=['resnet101', 'resnet50', 'seresnet50', 'xception',
                                                                'resnest50', 'resnest101', 'resnest200', 'resnest269'],
                        help='deeplab backbone')
    parser.add_argument('--backbone', default='resnet34', choices=['resnet101', 'resnet50', 'resner34', 'other'], help='dinknet backbone')
    parser.add_argument('--enhance', default='dblock', choices=['aspp', 'dblock'], help='deeplab mid layer')
    parser.add_argument('--modelname', default='deeplab', choices=['deeplab', 'lednet', 'hrnetv2', 'ocrnet', 'dinknet'])
    parser.add_argument('--root', default='/workspace/lyf/data/ccf2020/train_data/', help='train data path')
    parser.add_argument('--model_path', default='/workspace/lyf/data/ccf2020/train_data/save_model/', help='path to save checkpoint')
    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal', 'dice'])
    parser.add_argument('--device_ids', default=[0])
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer()


if __name__ == '__main__':
    main()
