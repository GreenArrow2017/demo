__package__ = 'run'
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import sys
from utils import draw_fig
from models_code.CNNModel import CNNModel
from models_code.ResNet import mnist_model
from torch.optim.lr_scheduler import StepLR
import numpy as np
# # nohup python mnist_cifar/run/ensemble_mnist.py >> output_mnist_ensemble_hyper4.log 2>&1 &
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

n_epochs = 150
batch_size_train = 240
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10
random_seed = 2
path = 'mnist_cifar/run/checkpoints/'
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            # 效果更加不好，我怀疑是增加了之后会使得很多图片变得有二义性，比如3一拉就变成5了， 1一拉长就变7了
            # torchvision.transforms.Resize(32),
            # torchvision.transforms.CenterCrop(28),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            torchvision.transforms.RandomRotation((-10, 10)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data',
                                                                     train=False,
                                                                     download=True,
                                                                     transform=torchvision.transforms.Compose(
                                                                         [torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])),
                                          batch_size=batch_size_test,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=True)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

models_num = 4

model_ensembles = [0] * models_num
optimizers = [0] * models_num
schedulers = [0] * models_num

train_losses = []
train_counter = []
train_acces = []
test_losses = []
test_acces = []
test_acces_agents = []
str_test_accs = []

'''
模型数量 = 4，一半用CNN，一半用RestNet
'''

for i in range(models_num):

    train_losses.append([])
    train_counter.append([])
    train_acces.append([])
    test_acces_agents.append([])

    if i % 2 == 0:
        # 这部分是CNN的
        model_ensembles[i] = CNNModel()
        model_ensembles[i].to(device)
        model_ensembles[i].apply(weight_init)
        optimizers[i] = optim.RMSprop(model_ensembles[i].parameters(), lr=learning_rate, alpha=0.99, momentum=momentum)
        schedulers[i] = lr_scheduler.ReduceLROnPlateau(optimizers[i], mode='max', factor=0.5, patience=3, verbose=True, threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    else:
        model_ensembles[i] = mnist_model()
        model_ensembles[i].to(device)
        model_ensembles[i].apply(weight_init)
        optimizers[i] = optim.Adam(model_ensembles[i].parameters(), lr=learning_rate, betas=(0.9, 0.99))
        schedulers[i] = StepLR(optimizers[i], step_size=5, gamma=0.1)

def train(network, optimizer, epoch, n_model):

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
            train_losses[n_model].append(loss.item())
            # 计数
            train_counter[n_model].append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

    train_acc = train_correct / len(train_loader.dataset)
    train_acces[n_model].append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))

def test(epoch):

    for n_model in range(models_num):
        model_ensembles[n_model].eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            output = torch.zeros(data.shape[0], 10)
            output = torch.tensor(output, dtype=torch.float16, device=device)


            for n in range(models_num):
                output = output + model_ensembles[n](data.to(device))
            output = output/models_num
            #test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()

            pred = output.data.max(dim=1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred).to(device)).sum() 
    acc = correct / len(test_loader.dataset)
    test_acces.append(acc.cpu().numpy().tolist())
    str_test_accs.append(str(acc.cpu().numpy().tolist()))

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    f = open("mnist_cifar/run/acc/test_acc_ensemble_hyper4.text", 'w')
    f.write('\n'.join(str_test_accs))
    f.close()


    # 打印相关信息 如：Test set: Avg. loss: 2.3129, Accuracy: 1205/10000 (12%)
    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n'\
          .format(epoch,test_loss, correct,len(test_loader.dataset),100. * acc),end = '')

for epoch in range(1, n_epochs+1):
    for n_model in range(models_num):
        index = n_model
        print(f"第{n_model}号：")
        train(network=model_ensembles[index], optimizer=optimizers[index], epoch=epoch, n_model=n_model)
        if n_model % 2 == 0:
            schedulers[index].step(train_acces[n_model][-1])
        else:
            schedulers[index].step()
    test(epoch=epoch)


print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))


f = open("mnist_cifar/run/acc/test_acc_ensemble_hyper4.text", 'w')
f.write('\n'.join(str_test_accs))
f.close()


