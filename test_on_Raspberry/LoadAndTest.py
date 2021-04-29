# -*- coding: utf-8 -*-
"""
Created on Thu April 20 11:10:40 2021

@author: Mingze Gong
"""

import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
import argparse 
from RDPNet import RDPNet
import time
import os
import psutil 

device = torch.device("cpu")
net = RDPNet(num_classes=10).to(device)
path = "./RDPNetmodel.pth"
net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

testset = torchvision.datasets.CIFAR10(root='../cifar10-data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

sum_loss = 0.00
test_loss = 0.00
epochs = 1
total_memory = 0
total_time = 0
for epoch in range (0, epochs) :                
    print("Testing--->")
    with torch.no_grad() :
        correct = 0
        total = 0
        num = 1
        for data in testloader :
            start = time.perf_counter() 
            net.eval()
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            end = time.perf_counter()
            print ("Seq: " + str(num))
            num += 1
            print ("Time Cost: %3f"%(float(end - start)))
            print('memory: %.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024) )
            total_memory = total_memory + float(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
            total_time = total_time + float(end-start)
        print ("Test Accuracy : %.3f%%" % (100 * correct / total))
   
print ("Finish!")
print ("Time Cost: %3f"%(total_time))
print ("Average Time Cost: %3f"%( (total_time / 100 )))
print ("Average Memory Usage: %3f"%(total_memory / 100))

with open ("record.txt","w") as file1:
    file1.write("Time Cost: %3f"%(total_time))
    file1.write("\n")
    file1.write("Average Time Cost: %3f"%( (total_time / 100 )))
    file1.write("\n")
    file1.write("Average Memory Usage: %3f"%(total_memory / 100))
    file1.close