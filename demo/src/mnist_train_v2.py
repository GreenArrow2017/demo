import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
import cv2
import os
from model import CNNModel, CNNModel_5, mnist_model
# /home/linyangkai/env/LBF_env/bin/python3.8 /home/linyangkai/projects/MNIST/src/mnist_train_v2.py
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()
#初始化变量
n_epochs = 100  #训练次数
batch_size_train = 240  #训练的 batch_size
batch_size_test = 1000  #测试的 batch_size
learning_rate = 0.001  # 学习率
momentum = 0.5  # 在梯度下降过程中解决mini-batch SGD优化算法更新幅度摆动大的问题，使得收敛速度更快
log_interval = 10  # 操作间隔
random_seed = 2  # 随机种子，设置后可以得到稳定的随机数
torch.manual_seed(random_seed)


#图片处理
def imageProcess(img):
    #处理图片
    data_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(32),
         torchvision.transforms.CenterCrop(28),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化
    fanse = cv2.bitwise_not(dst)  #黑白反转

    #将BGR图像转变成RGB图像：即将cv2.imread转换成Image.open
    imgs = Image.fromarray(cv2.cvtColor(fanse, cv2.COLOR_BGR2RGB))
    imgs = imgs.convert('L')  #将三通道图像转换成单通道灰度图像
    imgs = data_transform(imgs)  #处理图像

    return imgs


#导入训练集并增强数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './mnist/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            # 效果更加不好，我怀疑是增加了之后会使得很多图片变得有二义性，比如3一拉就变成5了， 1一拉长就变7了
            # torchvision.transforms.Resize(32),
            # torchvision.transforms.CenterCrop(28),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            torchvision.transforms.RandomRotation((-10, 10)),  #将图片随机旋转（-10,10）度
            torchvision.transforms.ToTensor(),  # 将PIL图片或者numpy.ndarray转成Tensor类型
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=4,
    pin_memory=True)  # shuffle如果为true,每个训练epoch后，会将数据顺序打乱
#导入测试集
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./mnist/',
                                                                     train=False,
                                                                     download=True,
                                                                     transform=torchvision.transforms.Compose(
                                                                         [torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])),
                                          batch_size=batch_size_test,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=True)


#权值初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    '''
    这两个加了效果不好，-0.2%了都
    '''
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight)
    #     nn.init.constant_(m.bias, 0)
    # # 是否为批归一化层
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)


# 实例化一个网络
network = CNNModel()
network.to(device)
#调用权值初始化函数
network.apply(weight_init)
# 设置优化器，用stochastic gradient descent，设置学习率，设置momentum
#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = optim.Adam(network.parameters(), lr=learning_rate)
# 用这个loss记得带softmax，如果是SGD和Adam可以不要
optimizer = optim.RMSprop(network.parameters(), lr=learning_rate, alpha=0.99, momentum=momentum)
#设置学习率梯度下降，如果连续三个epoch测试准确率没有上升，则降低学习率
# 这个真的是好东西
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
#定义存储数据的列表
train_losses = []
train_counter = []
train_acces = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
test_acces = []


# 定义训练函数
def train(epoch):

    network.train()  # 将网络设为 training 模式
    train_correct = 0
    # 对一组 batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # 通过enumerate获取batch_id, data, and label
        # 1-将梯度归零
        optimizer.zero_grad()

        # 2-传入一个batch的图像，并前向计算
        # data.to(device)把图片放入GPU中计算
        output = network(data.to(device))

        # 3-计算损失
        loss = F.nll_loss(output, target.to(device))

        # 4-反向传播
        loss.backward()

        # 5-优化参数
        optimizer.step()
        #exp_lr_scheduler.step()

        train_pred = output.data.max(dim=1, keepdim=True)[1]  # 取 output 里最大的那个类别,
        # dim = 1表示去每行的最大值，[1]表示取最大值的index，而不去最大值本身[0]

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # 比较并求正确分类的个数
        #打印以下信息：第几个epoch，第几张图像， 总训练图像数, 完成百分比，目前的loss
        print('\r 第 {} 次 Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()), end='')

        # 每第10个batch (log_interval = 10)
        if batch_idx % log_interval == 0:
            #print(batch_idx)
            # 把目前的 loss加入到 train_losses,后期画图用
            train_losses.append(loss.item())
            # 计数
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))


# 定义测试函数
def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device))
            #test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()

            pred = output.data.max(dim=1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred).to(device)).sum()  # 比较并求正确分类的个数
    acc = correct / len(test_loader.dataset)
    test_acces.append(acc.cpu().numpy().tolist())

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    #保存测试准确率最大的模型
    if test_acces[-1] >= max(test_acces):

        torch.save(network.state_dict(), './model03.pth')

        torch.save(optimizer.state_dict(), './optimizer03.pth')

    # 打印相关信息 如：Test set: Avg. loss: 2.3129, Accuracy: 1205/10000 (12%)
    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n'\
          .format(epoch,test_loss, correct,len(test_loader.dataset),100. * acc),end = '')


###################################################
# 第一轮训练，99.73%
# 根据epoch数正式训练并在每个epoch训练结束后测试
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)
    scheduler.step(test_acces[-1])
#输入最后保存的模型的准确率，也就是最高测试准确率
print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))

# 实例化一个网络
network = CNNModel()
#加载模型
model_path = "./model02.pth"
network.load_state_dict(torch.load(model_path))
network.to(device)

#network.apply(weights_init)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True, threshold=1e-06, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-09)

for epoch in range(101, 201):
    train(epoch)
    test(epoch)
    scheduler.step(train_acces[-1])
print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))
