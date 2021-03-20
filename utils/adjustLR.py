from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ExponentialLR,\
    CosineAnnealingLR, LambdaLR, CyclicLR
from torch.optim import Adam, SGD
from torchvision.models import AlexNet
import matplotlib.pyplot as plt


class AdjustLr:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def ReduceLROnPlateau_(self, mode='max', factor=0.8, patience=10):
        scheduler = ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience, verbose=True)
        return scheduler

    def StepLR_(self, step_size=10, gamma=0.8):
        scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return scheduler

    def MultiStepLR_(self, milestones=[10, 50, 80], gamma=0.8):
        scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        return scheduler

    def ExponentialLR_(self, gamma=0.8):
        scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        return scheduler

    def CosineAnnealingLR_(self, T_max=10, eta_min=0):
        scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        return scheduler

    def CosineAnnealingWarmRestarts_(self, T_0=10, T_mult=2):
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)
        return scheduler

    def LambdaLR_(self, milestone=5, gamma=0.98):
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: gamma ** (epoch - milestone) if (epoch > milestone) else 1.)
        return scheduler

    def CyclicLR_(self, base_lr=0.00001, max_lr=0.003, step_size_up=20, step_size_down=20, mode='triangular'):
        scheduler = CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
                             step_size_down=step_size_down, mode=mode)
        return scheduler

    def adjust(self, base_lr, type):
        pass


if __name__ == '__main__':
    net = AlexNet(num_classes=2)
    optimizer = SGD(net.parameters(), lr=0.0002)
    adj = AdjustLr(optimizer)
    sch1 = adj.LambdaLR_(milestone=5, gamma=0.92)
    epoches = 40
    plt.figure()
    x1 = list(range(epoches))
    y1 = list()
    for epoch in range(epoches):
        optimizer.step()
        sch1.step(epoch)

        a = sch1.get_lr()
        y1.append(optimizer.param_groups[0]['lr'])
    plt.plot(x1, y1)
    plt.show()