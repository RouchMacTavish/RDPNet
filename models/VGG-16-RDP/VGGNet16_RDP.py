# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:46:08 2021

@author: Mingze Gong 
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os 



class VGGNet16 (nn.Module) :
    def __init__(self, num_classes=2) :
        super (VGGNet16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.conv1_ = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.conv2_ = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            )
        self.conv3_ = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv4_ = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, num_classes),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, num_classes),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512,num_classes),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes)
        )


    def forward (self, x) :
        x = self.conv1(x)

        x1 = self.conv1_(x)
        x1 = F.max_pool2d(x1, 2)

        x = F.max_pool2d(x, 2)

        x1 = x1 - x

        x += x1

        x1 = F.max_pool2d(x1, 2)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)

        x = self.conv2(x)

        x2 = self.conv2_(x)
        x2 = F.max_pool2d(x2, 2)

        x = F.max_pool2d(x, 2)

        x2 = x2 - x

        x += x2

        x2 = F.max_pool2d(x2, 2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)

        x = self.conv3(x)

        x3 = self.conv3_(x)
        x3 = F.max_pool2d(x3, 2)

        x = F.max_pool2d(x, 2)

        x3 = x3 - x

        x += x3

        x3 = F.max_pool2d(x3, 2)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc3(x3)

        x = self.conv4(x)
        
        x4 = self.conv4_(x)
        x4 = F.max_pool2d(x4, 2)
        
        x = F.max_pool2d(x, 2)
        
        x4 = x4 - x

        x += x4

        x4 = F.max_pool2d(x4, 2)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.fc4(x4)

        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        
        x = 0.1*x1 + 0.3*x2 + 0.6*x3 + 0.7*x
        
        return x

