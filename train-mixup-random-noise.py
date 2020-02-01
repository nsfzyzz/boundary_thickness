import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms

from utils import *
from resnet_cifar import resnet20_cifar, resnet56_cifar
from models_kuangliu.resnet import ResNet18
from tqdm import tqdm, trange

import torchvision.utils
import matplotlib.pyplot as plt

import logging
import os
import sys

from torch.utils.data import TensorDataset
from mpl_toolkits.mplot3d import Axes3D


# Training settings
name = 'cifar10'
batch_size = 64
test_batch_size = 200
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_loader, test_loader = getData(name=name, train_bs=batch_size, test_bs=test_batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model = ResNet18()
model.linear = nn.Linear(in_features=512, out_features=11)
model = model.to(device)

model = torch.nn.DataParallel(model)
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

eval_model(model, test_loader, "original test images")

## Mixup training

criterion = nn.CrossEntropyLoss()

# Note that here, according to the original mixup repo, the weight_decay should be set smaller
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 1e-4)

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


num_epochs = 200
Loss_list = []
best_acc = 0.0

alpha = 1
use_cuda = torch.cuda.is_available()
total_step = len(train_loader)

for epoch in range(num_epochs):

    model.train()

    if epoch > num_epochs * 0.75:
        lr = 0.0001
    elif epoch > num_epochs * 0.5:
        lr = 0.001
    else:
        lr = 0.01

    update_lr(optimizer, lr)

    for i, (inputs, targets) in enumerate(train_loader):

        # generate random inputs
        inputs_random = torch.rand_like(inputs)
        # normalize
        inputs_random[:, 0, :, :] = (inputs_random[:, 0, :, :] - 0.4914) / 0.2023
        inputs_random[:, 1, :, :] = (inputs_random[:, 1, :, :] - 0.4822) / 0.1994
        inputs_random[:, 2, :, :] = (inputs_random[:, 2, :, :] - 0.4465) / 0.2010

        # noise is the 11-th class
        targets_random = torch.ones_like(targets) * 10.0

        inputs = torch.cat([inputs, inputs_random])
        targets = torch.cat([targets, targets_random])

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

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            Loss_list.append(loss.item())

    if (epoch + 1) % 5 == 0 or epoch <= 5:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        acc = eval_model(model, test_loader, 'src test data')

        if acc > 0:
            print('Saving model ' + '..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ResNet18_mixup_{}.pth'.format(epoch))
print('acc :', acc)

