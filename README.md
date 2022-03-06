# MNISTandCIFAR10
### MNIST
1. ResNet + Adam + StepLR : 99.70% (mnist_train.py)
2. ResNet + RMSprop + ReduceLROnPlateau : **99.76%** (mnist_train_v2.py)
### CIFAR10
1. ResNet + Weight Standardization[1]: 93.7%。(cifar10_train.py)
2. ResNet18 + pretrain + RandomHorizontalFlip : **97%** (cifar10_train_v2.py) (RandomVerticalFlip加了反而帮倒忙？)

#### [1] Qiao, Siyuan, et al. "Micro-batch training with batch-channel normalization and weight standardization." arXiv preprint arXiv:1903.10520 (2019). 

