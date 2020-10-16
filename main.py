import argparse
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation master")
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--batch_size', default=24)
    parser.add_argument('--scales', default=[0.8, 1.2])
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--size', default=512)
    parser.add_argument('--use_balanced_weights', default=False)
    parser.add_argument('--pretrain', default=None)
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--best_miou', default=0.0)
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--output_stride', default=16, help='deeplab os')
    parser.add_argument('--arch', default='resnet50', choices=['resnet101', 'resnet50', 'seresnet50'], help='deeplab backbone')
    parser.add_argument('--backbone', default='resnet34', choices=['resnet101', 'resnet50', 'resner34', 'other'], help='dinknet backbone')
    parser.add_argument('--enhance', default='aspp', choices=['aspp', 'dblock'], help='deeplab mid layer')
    parser.add_argument('--modelname', default='dinknet', choices=['deeplab', 'lednet', 'hrnetv2', 'ocrnet', 'dinknet'])
    parser.add_argument('--root', default='/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/data/ljl/', help='train data path')
    parser.add_argument('--model_path', default='/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/save_model/', help='path to save checkpoint')
    parser.add_argument('--loss_type', default='dice', choices=['ce', 'focal', 'dice'])
    parser.add_argument('--device_ids', default=[0])
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer()


if __name__ == '__main__':
    main()
