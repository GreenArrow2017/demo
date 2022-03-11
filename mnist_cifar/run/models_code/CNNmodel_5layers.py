#引用需要用到的库
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
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution layer 1
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 64, kernel_size = 5, stride = 1, padding = 2 )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels =64 , out_channels = 64, kernel_size = 5, stride = 1, padding = 2 )
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.drop1 = nn.Dropout(0.25)

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1 )
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1 )
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.drop2 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1 )
        self.relu5 = nn.ReLU()
        self.batch5 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout(0.25)
        
        # Fully-Connected layer 1
        
        self.fc1 = nn.Linear(3136,256)
        self.fc1_relu = nn.ReLU()
        self.batch5 = nn.BatchNorm2d(64)
        self.dp1 = nn.Dropout(0.25)
        
        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,10)
        
                
    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)
        
        out = self.maxpool1(out)
        out = self.drop1(out)

 
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)
        
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.batch4(out)
        
        out = self.maxpool2(out)
        out = self.drop2(out)
        
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.batch5(out)
        out = self.drop3(out)


        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dp1(out)
        
        out = self.fc2(out)

        return F.log_softmax(out,dim = 1)