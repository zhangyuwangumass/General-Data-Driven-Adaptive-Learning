from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset
import torch.cuda as cutorch

import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable

from trajectoryPlugin.plugin import API


# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
num_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

'''



full_idx = np.arange(10000)
valid_idx = np.random.choice(10000, 500, replace=False).tolist()
test_idx = np.delete(full_idx, valid_idx).tolist()

#valid_sampler = SubsetRandomSampler(valid_idx)
#test_sampler = SubsetRandomSampler(test_idx)

validset = Subset(testset, valid_idx)
testset = Subset(testset, test_idx)


validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


api = API(num_cluster=3, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), iprint=2)
api.dataLoader(trainset, validset)
'''


for batch_idx, (inputs, targets) in enumerate(trainloader):
    img = inputs[0].permute(1,2,0)
    print(img.shape)
    plt.imshow(img)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    break


img = trainset.train_data[0]
print(img.shape)
plt.imshow(img)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()
'''
for batch_idx, (inputs, targets) in enumerate(testloader):
    img = inputs[0]
    img = img.permute(1,2,0)
    print(img.shape)
    plt.imshow(img)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    break
'''