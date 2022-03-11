# MNISTandCIFAR10
### MNIST
1. ResNet + Adam + StepLR : 99.70% 
2. CNN4层卷积 + RMSprop + ReduceLROnPlateau : **99.75%**  (mnist_9975.py)
3. CNN5层卷积 + RMSprop + ReduceLROnPlateau : **99.77%**  (mnist_9977.py)
4. ensemble 5 个CNN +　RMSprop + ReduceLROnPlatea：**99.75%**
5. ensemble 4 个模型，2个resnet+2个CNN : **99.78%**(ensemble_mnist_9978.py)
### CIFAR10
1. ResNet + Weight Standardization[1]: 93.7%。
2. ResNet18 + pretrain + RandomHorizontalFlip : **97%** (cifar10.py) (RandomVerticalFlip加了反而帮倒忙？)
### 文件目录
##### 1.model_code: 包含所有模型
##### 2.acc和loss: 准确率和Loss的曲线
##### 3.pic: 准确率的截图
##### 4.augment.py: 数据集增强，cifar.py: cifar10代码，mnist.py: mnist代码，utils.py: 画图代码

### acc文件夹
##### test_acc_cifar.text，Train_acc_cifar.png ：cifar的ResNet18准确率和曲线
##### test_acc_mnist.text，Train_acc_mnist_single.png：mnist的CNN层准确率和曲线
##### test_acc_ensemble_5.text，Train_acc_mnist_ensemble_5.png : mnist的aggregation CNN的准确率和曲线
##### test_acc_ensemble_hyper4.text，Train_acc_mnist_ensemble_hyper4.png ：mnist的aggregation CNN+resnet的准确率和曲线
