import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import  datasets,transforms,utils
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
train_data = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

test_data = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
train_data = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)

test_data = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=4,
                                          shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=4,
                                          shuffle=True,num_workers=2)

#模型定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1=nn.Linear(64*7*7,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))

        x=x.view(-1,64*7*7)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


#模型训练
train_accs=[]
train_loss=[]
test_accs=[]

device=torch.device('cuda:2' if torch.cuda.is_available() else "cpu")

net= CNN()
net= net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):
    running_loss=0.0
    for i,data in enumerate(train_loader,0):
        inputs,labels=data[0].to(device),data[1].to(device)
        # print(labels)
        optimizer.zero_grad()

        outputs=net(inputs)

        # print(outputs.size)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        #打印损失值
        if i%100 ==99:
            print('[%d,%5d] loss :%.3f'%(epoch+1,i+1,running_loss/100))
            running_loss=0.0
        train_loss.append(loss.item())

        #计算精确率
        correct=0
        total=0
        _,predicted=torch.max(outputs.data,1)
        total=labels.size(0)
        correct=(predicted==labels).sum().item()
        train_accs.append(100*correct/total)

print('Finished Training')


