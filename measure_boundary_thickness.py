from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable

from resnet_cifar import resnet56_cifar

from attack_functions import *
from utils import *
from tqdm import tqdm, trange

import logging
import os
import sys

from torch.utils.data import TensorDataset


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_bs=128
test_bs=128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
            
model = resnet56_cifar().cuda()
model = torch.nn.DataParallel(model)
file_path = "../checkpoint/resnet56_cifar_300.pkl"
model.load_state_dict(torch.load(file_path))

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print("Finished loading {0}".format(file_path))

train_loader, test_loader = getData(name="cifar10", train_bs=train_bs, test_bs=test_bs)

test(model, test_loader)

# The following is for measuring the boundary thickness

ave_boundary_thickness = 0
num_lines = 0

softmax1 = nn.Softmax()

for i, (data, labels) in enumerate(train_loader):
    
    if i>300:
        break

    model.eval()

    data, labels = data.cuda(), labels.cuda()

    output = model(data)
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

    # random sample until we get two data points that have different labels

    while True:
        ind1 = random.randint(0, pred.shape[0]-1)
        ind2 = random.randint(0, pred.shape[0]-1)

        if pred[ind1] != pred[ind2]:

            break

    ## Sample 128 points from these two data points

    x1, x2 = data[ind1], data[ind2]
    dist = torch.norm(x1 - x2, p=2)
    new_batch = []

    num_points = 128
    for lmbd in np.linspace(0, 1.0, num=num_points):
        new_batch.append(x1*lmbd + x2*(1-lmbd))
    new_batch = torch.stack(new_batch)

    model.eval()

    y_new_batch = softmax1(model(new_batch))
    y_new_batch = y_new_batch[:, pred[ind1]].detach().cpu().numpy().flatten()
    #print(y_new_batch)
    #print(y_new_batch.shape)
    
    # The 0.25 and 0.75 here can be changed to some other numbers
    boundary_thickness = np.logical_and((0.75 > y_new_batch), (0.25 < y_new_batch))
    boundary_thickness = dist * np.sum( boundary_thickness )/num_points

    ave_boundary_thickness += boundary_thickness
    num_lines += 1

print("The boundary thickness is {0}".format(float(ave_boundary_thickness)/num_lines))

