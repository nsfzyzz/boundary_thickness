from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from resnet_cifar import resnet20_cifar
from resnet_cifar import resnet56_cifar
from tqdm import tqdm, trange

import torchvision.utils
import matplotlib.pyplot as plt

import logging
import os
import sys

from torch.utils.data import TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# In[34]:


# Training settings
name = 'cifar10'
batch_size = 128
test_batch_size = 200
seed = 1
arch = 'resnet_large'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_loader, test_loader = getData(name=name, train_bs=batch_size, test_bs=test_batch_size)

model_list = {
    'resnet': resnet20_cifar(),
    'resnet_large': resnet56_cifar(),
    'alex_net': Alex_Net()
}

model = model_list[arch].cuda()
model = torch.nn.DataParallel(model)
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

def eval_model(model_tst, loader_tst, dataset_name):

    print("Test the model")

    model_tst.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader_tst:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_tst(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(total)

        print('Test Accuracy of the model on the ' + dataset_name + ' : {} %'.format(100 * correct / total))
        
    return 100 * correct / total


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

eval_model(model, test_loader, "original test images")

## Mixup training

criterion = nn.CrossEntropyLoss()

# Note that here, according to the idea of mixup, the weight_decay should be set smaller
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 1e-4)  

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

num_epochs = 1000
Loss_list = []
best_acc = 0.0
train = True
#train = False
alpha = 1
use_cuda = torch.cuda.is_available()
total_step = len(train_loader)

if train :
    
    for epoch in range(num_epochs):
        
        model.train()
        
        if epoch > 450:
            lr = 0.0001
        elif epoch > 300:
            lr = 0.001
        else:
            lr = 0.01

        update_lr(optimizer, lr)
        
        for i, (inputs, targets) in enumerate(train_loader):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            # generate mixed inputs, two one-hot label vectors and mixing coefficient
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, use_cuda)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                Loss_list.append(loss.item())

        if (epoch+1) % 5 == 0 or epoch<=5:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

            acc = eval_model(model,test_loader ,'src test data')

            if acc > 0:
                print('Saving model ' + '..')
                state = {
                    'net': model.state_dict(),
                    'acc': acc,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/resnet_cifar_mixup_large_{}.pth'.format(epoch))
    print('acc :' , acc)

