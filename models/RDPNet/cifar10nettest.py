# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:10:40 2021

@author: Mingze Gong
"""

import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
import argparse 
#from torch.autograd import Variable
#from resnet import Resnet18
#from TestNet import PoDoGNet
from RDPNet import RDPNet
#from CommonNet import CommonNet
#from VGGNet11 import VGGNet11
#from VGGNet11_RDP import VGGNet11
#from VGGNet16 import VGGNet16
#from VGGNet16_RDP import VGGNet16
#from VGGNet19 import VGGNet19
#from VGGNet19_RDP import VGGNet19
import os 
import time 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 150
pre_epoch = 0
batch_size = 128
lr = 0.01

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    #transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_test = transforms.Compose([
    #transforms.Resize((64,64)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data1', train=True, download=True, transform=transform_train )
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data1', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#net = PoDoGNet(num_classes=10).to(device)
net = RDPNet (num_classes=10).to(device)
#net = VGGNet11(num_classes=10).to(device)
#net = VGGNet16(num_classes=10).to(device)
#net = VGGNet19(num_classes=10).to(device)
#net = CommonNet(num_classes=10).to(device)
#net = Resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) #5e-4 1e-3
#optimizer = optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

best_acc = 85
print ("Start Training !")
start = time.perf_counter()
with open ("acc_RDPnet.txt","w") as file1 :
    with open ("log_RDPnet.txt","w") as file2 :
        for epoch in range (pre_epoch, epochs) :
            print ("\nEpoch : " + str(epoch+1))
            print ("Training--->")
            start_train = time.perf_counter()
            net.train()
            sum_loss = 0.00
            test_loss = 0.00
            correct = 0.00
            total = 0.00
            for i, data in enumerate(trainloader,0) :
                length = len(trainloader)
                inputs , labels = data
                inputs , labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                end_train = time.perf_counter() 
                print ('[epoch:%d, iter:%d] Loss: %.05f | Acc: %.3f%% |'
                       % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                file2.write('%03d  %05d |Loss: %.05f | Acc: %.3f%% |'
                         % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                file2.write("\n")
                file2.flush()
                
            print("Testing--->")
            with torch.no_grad() :
                correct = 0
                total = 0
                start_test = time.perf_counter() 
                for data in testloader :
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_loss = criterion(outputs, labels)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                
                end_test = time.perf_counter()
                print ("Test Accuracy : %.3f%%" % (100 * correct / total))
                acc = 100.* correct / total
                
                file1.write("Epoch=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                file1.write(", Loss:%.3f" % (test_loss))
                file1.write(", Test Time: %3f" %(float(end_test - start_test)))
                file1.write("\n")
                file1.flush()
                
                if acc > best_acc :
                    if acc > 91 :
                        print ("Saving Model--->")
                        torch.save(net.state_dict(), 'RDPnetmodel'+str(epoch+1)+'.pth')
                    file3 = open("best_acc_RDPnet.txt","a")
                    file3.write ("Epoch=%d, best_acc=%.3f%%" % (epoch+1, acc))
                    file3.write ("\n")
                    file3.close()
                    best_acc = acc 
        print ("Training Complete! Total_Epoch = %d" % epochs)
            

end = time.perf_counter() 
time_out = end - start
with open ("time_costRDP.txt","w") as filet :
    filet.write("Total Time Cost : " + str(int(time_out)) + "s")
    filet.write("\n")
    filet.write("Average Time Cost : " + str(int(time_out/epochs)) + "s")
    filet.close