# 

import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import pandas as pd
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 数据增强
train_transforms = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 数据集
train_dataset = CIFAR10(root='data', train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10(root='data', train=False, download=True, transform=test_transforms)

sample_index = [i for i in range(500)] #假设取前500个训练数据
X_train = []
y_train = []
for i in sample_index:
    X = train_dataset[i][0]
    output_tensor = F.interpolate(X.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)  
    #print(output_tensor.shape)
    X_train.append(output_tensor)
    y = train_dataset[i][1]
    y_train.append(y)

sampled_train_data = [(X, y) for X, y in zip(X_train, y_train)] #包装为数据对
#trainDataLoader = torch.utils.data.DataLoader(sampled_train_data, batch_size=16, shuffle=True)

# DataLoader
train_loader = torch.utils.data.DataLoader(sampled_train_data, batch_size=64, shuffle=True, num_workers=4)

sample_index = [i for i in range(500,1000)] #假设取500-1000个训练数据
X_test = []
y_test = []
for i in sample_index:
    X = test_dataset[i][0]
    output_tensor = F.interpolate(X.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)  
    #print(output_tensor.shape)
    X_test.append(output_tensor)
    y = test_dataset[i][1]
    y_test.append(y)

sampled_test_data = [(X, y) for X, y in zip(X_test, y_test)] #包装为数据对
test_loader = torch.utils.data.DataLoader(sampled_test_data, batch_size=64, shuffle=False, num_workers=4)

# 加载预训练模型
model = timm.create_model('resnet18', pretrained=True)

# 修改分类器
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # 添加全连接层：降维1000-10，匹配任务


model=model.to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10

for epoch in range(num_epochs):
    time_start=time.time()
    # 训练
    model.train()
    for images, labels in train_loader:
        # 前向传播
        images = images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i=i+1
        if i%10==0:
            print("-",end=' ')
    # 测试
    time_train=time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        time_eval=time.time()
        print('Epoch {} Accuracy: {:.2f}%'.format(epoch+1, 100*correct/total))
        print("train_time= {:.2f}s; eval_time= {:.2f}s".format(time_train-time_start,time_eval-time_train))
