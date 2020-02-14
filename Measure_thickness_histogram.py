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

from models_kuangliu.resnet_without_link import ResNet50 as ResNet50_without_link
from models_kuangliu.resnet_with_link import ResNet50 as ResNet50_with_link
from models_kuangliu.resnet import ResNet18
from models_kuangliu.resnet_without_link import ResNet18 as ResNet18_without_link
from models_kuangliu.densenet import densenet_cifar
from models_kuangliu.vgg import *
from resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet110_cifar

from attack_functions import *
from utils import *
from visualize_functions import *
from madry_model import WideResNet
from tqdm import tqdm, trange

import logging
import os
import sys

from torch.utils.data import TensorDataset

from model_zoo_new import *

parser = argparse.ArgumentParser(description='Measure boundary thickness')
parser.add_argument('--specific_class', type=int, default = -1, help='The specific class of image that we want to visualize (-1 means we randomly pick a class)')

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))
    
# In[2]:


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# In[3]:


train_bs=128
test_bs=32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

results = {}

softmax1 = nn.Softmax()

for exp_ind in range(32):

    if exp_ind not in models_in_training["exp_ids"]:
        continue
        
    print("Experiement {0}: {1}".format(exp_ind, exp_name_list[exp_ind]))

    transform_train, transform_test = return_transforms(data_type_list[exp_ind])

    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    print("Start checking the results in all epochs!!!")

    epoch_list = return_epoch_list(exp_ind)

    hist_list_epochs = {}

    for epoch_num in epoch_list:

        print("Experiement {0}: epoch-{1}".format(exp_ind, epoch_num))

        model = model_list[exp_ind].cuda()
        model = torch.nn.DataParallel(model)
        resume = loc_epoch(exp_ind, epoch_num)
        load_model = switcher.get(exp_ind, "nothing")
        load_model(model, resume)
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print("Finished loading {0}".format(resume))

        # We generate one line of data from each batch of the original data
        
        temp_hist = []
        
        for data, labels in train_loader:
            
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

            margin_thickness = np.logical_and((0.75 > y_new_batch), (0.25 < y_new_batch))
            margin_thickness = dist * np.sum( margin_thickness )/num_points
            
            temp_hist.append(margin_thickness)

        hist_list_epochs[epoch_num] = temp_hist
        
    results[exp_ind] = {"exp_ind": exp_ind, "name": exp_name_list[exp_ind], 
                        "epoch_list":epoch_list, "results":hist_list_epochs}
    
import pickle
f = open("results/thickness_hist.pkl","wb")
pickle.dump(results, f)
f.close()

