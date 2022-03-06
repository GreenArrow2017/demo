from dataloader import CIFAR10_dataloader, MNIST_dataloader
from model import CNNmodel, MnistNet, mnist_model, densenet_cifar, PreActResNet18
from weightStdConv import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import Augmentor
from torch.optim.lr_scheduler import StepLR
from utils import *
import torch.backends.cudnn as cudnn

losses = []
accs = []


def train(model, optimizer, trainloader, testloader, criterion, device='cuda:1'):
    model.train()

    for index, (data, target) in tqdm(enumerate(trainloader), total=len(trainloader)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if index % 1000 == 0:
            acc = test(model, testloader, criterion)
            # adjust_learning_rate(optimizer, acc)


def test(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(testloader, total=len(testloader)):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
    global losses
    global accs
    losses.append(test_loss.cpu())
    accs.append(round(correct.item() / len(testloader.dataset), 2))
    return round(correct.item() / len(testloader.dataset), 2)


if __name__ == '__main__':
    '''
    ResNet(mnist_model): 93%
    '''

    device = 'cuda:0'
    n_epochs = 200
    batch_size_train = 128
    batch_size_test = 100
    learning_rate = 1e-1  # 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    nesterov = True
    weight_decay = 5e-4
    IMG_MEAN = [125.3, 123.0, 113.9]
    IMG_STD = [63.0, 62.1, 66.7]

    #=============================== data augmentation augmented_pipline比较好
    transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    trans = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation((-10, 10)),  # 将图片随机旋转（-10,10）度
        transforms.ToTensor(),  # 将PIL图片或者numpy.ndarray转成Tensor类型
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    #===============================
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #===============================

    trainloader = CIFAR10_dataloader(train=True, batch_size=batch_size_train, transform=transform_train, shuffle=True)
    testLoader = CIFAR10_dataloader(train=False, batch_size=batch_size_test, transform=transform_test, shuffle=False)

    model = resnet18(num_classes=10)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    # mnist_model: 93%
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')
        train(model, optimizer, trainloader, testLoader, criterion, device)
        scheduler.step()
    draw_fig(losses, 'loss', len(losses))
    draw_fig(accs, 'acc', len(accs))
