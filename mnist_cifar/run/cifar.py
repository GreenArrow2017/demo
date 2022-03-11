# coding=UTF-8
import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm 
from augment import flip_transform
from utils import draw_fig
from torch.utils.tensorboard import SummaryWriter

# nohup python mnist_cifar/run/cifar.py >> output_cifar.log 2>&1 &

myWriter = SummaryWriter('log/')
n_epochs = 60
accs = []
losses = []
Transforms = flip_transform()

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=Transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=Transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)

# myModel = torchvision.models.resnet101(pretrained=True)
myModel = torchvision.models.resnet50(pretrained=True)
# myModel = torchvision.models.resnet152(pretrained=True)
# 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
inchannel = myModel.fc.in_features
myModel.fc = nn.Linear(inchannel, 10)

myDevice = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)
print(myDevice)

learning_rate = 0.001
# SGD比Adam效果好，Adam一开始会很快，但是acc没有SGD高
myOptimzier = optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(n_epochs):
    training_loss = 0.0
    for _step, input_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)
        predict_label = myModel.forward(image)

        loss = myLoss(predict_label, label)

        myWriter.add_scalar('training loss', loss, global_step=_epoch * len(train_loader) + _step)

        myOptimzier.zero_grad()
        loss.backward()
        myOptimzier.step()

        training_loss = training_loss + loss.item()
    print(f"Avg Training Loss : {training_loss/len(train_loader)}")
    loss_t = training_loss/len(train_loader)
    losses.append(loss_t)
    correct = 0
    total = 0
    myModel.eval()
    for images, labels in tqdm(test_loader, total=len(test_loader)):
        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = myModel(images)
        numbers, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    accs.append(str(acc))
    print('Testing Accuracy : %.3f %%' % (100 * correct / total))
    myWriter.add_scalar('test_Accuracy', 100 * correct / total)
    f = open("mnist_cifar/run/acc/test_acc_cifar.text", 'w')
    f.write('\n'.join(accs))
    f.close()
    
