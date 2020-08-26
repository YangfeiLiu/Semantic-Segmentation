from train import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser(description="deeplabv3plus")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--use_balanced_weights', default=True)
    parser.add_argument('--pretrain', default=None)
    parser.add_argument('--best_miou', default=0.0)
    parser.add_argument('--start_epoch', default=0)
    parser.add_argument('--arch', default='resnet50', choices=['resnet101', 'resnet50', 'seresnet50'], help='deeplab backbone')
    parser.add_argument('--modelname', default='lednet', choices=['deeplab', 'lednet'])
    parser.add_argument('--root', default='/media/hp/1500/liuyangfei/tianzhidata/地表分类/训练样本数据/train_data1/', help='train data path')
    parser.add_argument('--model_path', default='/media/hp/1500/liuyangfei/tianzhidata/地表分类/训练样本数据/model/', help='path to save checkpoint')
    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal'])
    parser.add_argument('--device_ids', default=[0, 1])
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer()


if __name__ == '__main__':
    main()