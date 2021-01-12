## 语义分割算法收集
#### 加载数据：指定数据存放的位置，数据应具有以下结构 
#### data
       - image
         - 1.png
         - 2.png
         - ...
       - label
         - 1.png
         - 2.png
         - ...
       - train.txt 存放训练数据文件名
       - valid.txt 存放验证数据文件名
#### 已存在算法
DeepLab v3+  
backbone: resnet101, xception, resnet50, seresnet50, resnest系列

D-linknet \
backbone: resnet101, resnet50, resnet34

LEDNet

HRnetv2(输出增加了上采样)

OCRNet
#### 已存在损失函数
cross-entropy \
focal-loss \
dice-loss \
ocr-loss(没有用过)
