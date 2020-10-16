import argparse
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation master")
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--use_balanced_weights', default=True)
    parser.add_argument('--pretrain', default=None)
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--best_miou', default=0.0)
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--output_stride', default=16, help='deeplab os')
    parser.add_argument('--arch', default='seresnet50', choices=['resnet101', 'resnet50', 'seresnet50'], help='deeplab backbone')
    parser.add_argument('--backbone', default='resnet34', choices=['resnet101', 'resnet50', 'resner34', 'other'], help='dinknet backbone')
    parser.add_argument('--modelname', default='ocrnet', choices=['deeplab', 'lednet', 'hrnetv2', 'ocrnet', 'dinknet'])
    parser.add_argument('--root', default='/media/hp/1500/liuyangfei/data/road_data/DeepGlobe/', help='train data path')
    parser.add_argument('--model_path', default='/media/hp/1500/liuyangfei/model/road/', help='path to save checkpoint')
    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal'])
    parser.add_argument('--device_ids', default=[0])
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer()


if __name__ == '__main__':
    main()
