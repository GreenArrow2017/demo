import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
myWriter = SummaryWriter('log/')
n_epochs = 50


Transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    # 神了，加上下翻转有效果，加水平翻转就不行？？？ vertical直接掉0.2%了
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])



train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=Transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=Transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)



# resnet 101跑不起来
# myModel = torchvision.models.resnet101(pretrained=True)
myModel = torchvision.models.resnet50(pretrained=True)
# myModel = torchvision.models.resnet152(pretrained=True)
# 最后一层用一个fc效果不如两个好了
inchannel = myModel.fc.in_features
myModel.fc = nn.Linear(inchannel, 10)

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

# learning_rate = 0.01
learning_rate=0.001
myOptimzier = optim.SGD(myModel.parameters(), lr = learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(n_epochs):
    training_loss = 0.0
    for _step, input_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)  
        predict_label = myModel.forward(image)
       
        loss = myLoss(predict_label, label)

        myWriter.add_scalar('training loss', loss, global_step = _epoch*len(train_loader) + _step)

        myOptimzier.zero_grad()
        loss.backward()
        myOptimzier.step()

        training_loss = training_loss + loss.item()
    print(f"Avg Training Loss : {training_loss/len(train_loader)}")
    correct = 0
    total = 0
    myModel.eval()
    for images,labels in tqdm(test_loader, total=len(test_loader)):
        images = images.to(myDevice)
        labels = labels.to(myDevice)     
        outputs = myModel(images) 
        numbers,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
    myWriter.add_scalar('test_Accuracy',100 * correct / total)
