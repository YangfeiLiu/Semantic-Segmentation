from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ExponentialLR,\
    CosineAnnealingLR, LambdaLR, CyclicLR
from torch.optim import Adam, SGD
from torchvision.models import AlexNet
import matplotlib.pyplot as plt


class AdjustLr:
    def __init__(self, optimizer, metric='loss'):
        mode = 'min' if metric == 'loss' else 'max'
        self.ReduceLROnPlateau = ReduceLROnPlateau(optimizer, mode, factor=0.8, patience=10, verbose=True)
        self.StepLR = StepLR(optimizer, step_size=30, gamma=0.8)
        self.MultiStepLR = MultiStepLR(optimizer, milestones=[20, 65, 90], gamma=0.5)
        self.ExponentialLR = ExponentialLR(optimizer, gamma=0.98)
        self.CosineAnnealingLR = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)  # T_max是cos函数的半周期
        self.CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)  # T_0是第一次重启的周期,T_mult是周期增大倍数
        self.MyScheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.98**(epoch - 20) if (epoch > 20) else 1)
        # self.CyclicLR = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.003, step_size_up=20, step_size_down=20, mode='triangular')

    def adjust(self, base_lr, type):
        pass


if __name__ == '__main__':
    net = AlexNet(num_classes=2)
    optimizer = SGD(net.parameters(), lr=0.05)
    adj = AdjustLr(optimizer)
    sch1 = adj.CyclicLR
    plt.figure()
    x1 = list(range(100))
    y1 = list()
    for epoch in range(100):
        optimizer.step()
        sch1.step()

        a = sch1.get_lr()
        y1.append(sch1.get_lr()[0])
    plt.plot(x1, y1)
    plt.show()