import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Residual(nn.Module):
    
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm2d(d)
        self.conv3x3 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        return x + F.relu(self.conv3x3(x))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def mnist_model(d1=128, d2=256, d3=512):
    return nn.Sequential(
               nn.Conv2d(in_channels=1, out_channels=d1, kernel_size=5, padding=2),
               nn.ReLU(),

               Residual(d1),
               nn.MaxPool2d(2),
               Residual(d1),

               nn.BatchNorm2d(d1),
               nn.Conv2d(d1, d2, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               Residual(d2),

               nn.BatchNorm2d(d2),
               nn.Conv2d(d2, d3, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2, ceil_mode=True),
               Residual(d3),

               nn.BatchNorm2d(d3),
               nn.AvgPool2d(kernel_size=4),
               Flatten(),

               nn.Linear(d3,10),
               # Softmax provided during training.
               nn.LogSoftmax(dim=1)
           )