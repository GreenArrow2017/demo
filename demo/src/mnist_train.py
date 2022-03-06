from dataloader import CIFAR10_dataloader, MNIST_dataloader
from model import CNNmodel, MnistNet, mnist_model
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
from weightStdConv import *

losses = []
accs = []


def train(model, optimizer, scheduler, trainloader, testloader, criterion, device='cuda:1'):
    model.train()
    model = model.to(device)
    for index, (data, target) in tqdm(enumerate(trainloader), total=len(trainloader)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if index % 1000 == 0:
            test(model, testloader)


def test(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(testloader, total=len(testloader)):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
    global losses
    global accs
    losses.append(test_loss)
    accs.append(round(correct.item() / len(testloader.dataset), 2))


if __name__ == '__main__':
    '''
    MNIST：到目前为止最好的结构是99.7%
    参数：batch_size = 64
    学习率：4e-4，没三轮一次gamma=0.1衰减
    softmax: 不加softmax效果更好
    NetWork: ResNet
    '''

    seed = 66
    np.random.seed(seed)
    device = 'cuda:0'
    n_epochs = 20
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 4e-4
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.manual_seed(random_seed)

    #=============================== data augmentation augmented_pipline比较好
    transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    trans = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation((-10, 10)),  # 将图片随机旋转（-10,10）度
        transforms.ToTensor(),  # 将PIL图片或者numpy.ndarray转成Tensor类型
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    def img_normalize(t):
        c, x, y = t.shape
        t = t.view(c, -1)
        t = t - t.mean(dim=1, keepdim=True)
        t = t / t.std(dim=1, keepdim=True, unbiased=False)
        return t.view(c, x, y)

    def augmented_pipeline():
        rotate_distort = Augmentor.Pipeline()
        rotate_distort.random_distortion(1.0, 4, 4, magnitude=1)
        rotate_distort.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)

        cropsize = 25
        noncentered_crops = Augmentor.Pipeline()
        noncentered_crops.crop_by_size(1.0, cropsize, cropsize, centre=False)
        noncentered_crops.resize(1.0, 28, 28)

        return transforms.Compose([noncentered_crops.torch_transform(), rotate_distort.torch_transform(), transforms.ToTensor(), transforms.Lambda(img_normalize)])

    def nonaugmented_pipeline():
        centered_crops = Augmentor.Pipeline()
        cropsize = 25
        centered_crops.crop_by_size(1.0, cropsize, cropsize, centre=True)
        centered_crops.resize(1.0, 28, 28)

        return transforms.Compose([centered_crops.torch_transform(), transforms.ToTensor(), transforms.Lambda(img_normalize)])

    #===============================

    trainloader = MNIST_dataloader(train=True, batch_size=batch_size_train, transform=augmented_pipeline(), shuffle=True)
    testLoader = MNIST_dataloader(train=False, batch_size=batch_size_test, transform=nonaugmented_pipeline(), shuffle=False)

    # trainloader = CIFAR10_dataloader(train=True, batch_size=batch_size_train, transform=augmented_pipeline(), shuffle=True)
    # testLoader = CIFAR10_dataloader(train=False, batch_size=batch_size_test, transform=nonaugmented_pipeline(), shuffle=False)

    model = mnist_model()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        train(model, optimizer, scheduler, trainloader, testLoader, criterion, device)
        scheduler.step()

    draw_fig(losses, 'loss', len(losses))
    draw_fig(accs, 'acc', len(accs))
