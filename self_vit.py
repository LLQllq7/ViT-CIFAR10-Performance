import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import time
from Transformer import *
import copy
import time

TRAIN_SIZE=min(1000,50000)  # 训练集总个数50000
TEST_SIZE=min(1000,10000)   # 测试集总个数10000
print(f"TRAIN_SIZE = {TRAIN_SIZE}")

h=8
d_model=768
d_ff=3072
dropout=0.1
N=6
c = copy.deepcopy
attn = MultiHeadedAttention(h, d_model)
# attn是多头注意力,h是头数,d_model是词向量维度
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#ff是前馈神经网络，d_model是词向量维度，d_ff是前馈神经网络的维度,
#全连接层，先变长再变短。
position = PositionalEncoding(d_model, dropout)
#position是位置编码，d_model是词向量维度

# encoder 处理过程
model_en = nn.Sequential(
    nn.Conv2d(3,768,kernel_size=(16,16),stride=(16,16)),    #[N,3,224,224]->[N,768,14,14]
    nn.Flatten(start_dim=2),    #N[768,14,14]->[N,768,196]  
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), #[N,768,196]->[N,196,768]->[N,197,768]->[N,768]
    nn.Linear(768,10),  #
    nn.Softmax(dim=1)   # [0,1]
)

for p in model_en.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
#print(model_en)


# 图像分类的数据增强
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
sample_index = [i for i in range(TRAIN_SIZE)] #假设取前500个训练数据
X_train = []
y_train = []
for i in sample_index:
    X = train_dataset[i][0]
    output_tensor = F.interpolate(X.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)  # 将图片转化成224*224格式
    #print(output_tensor.shape)
    X_train.append(output_tensor)
    y = train_dataset[i][1]
    y_train.append(y)

sampled_train_data = [(X, y) for X, y in zip(X_train, y_train)] #包装为数据对
#trainDataLoader = torch.utils.data.DataLoader(sampled_train_data, batch_size=16, shuffle=True)

# DataLoader
train_loader = torch.utils.data.DataLoader(sampled_train_data, batch_size=64, shuffle=True, num_workers=4)

sample_index = [i for i in range(TEST_SIZE)] #假设取500-1000个训练数据
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

# 以上数据处理
# 以下模型训练

model=model_en
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
    i=0
    for images, labels in train_loader:     # 每 batch = 64 批处理
        # 前向传播
        images = images.to(device)
        labels=labels.to(device)
        outputs = model(images)         # 输出，0~9的概率
        #print(outputs.shape)
        #print(images.shape)
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
