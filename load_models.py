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

from models_kuangliu.resnet_without_link import ResNet50 as ResNet50_without_link
from models_kuangliu.resnet_with_link import ResNet50 as ResNet50_with_link
from models_kuangliu.resnet import ResNet18
from models_kuangliu.resnet_without_link import ResNet18 as ResNet18_without_link
from models_kuangliu.densenet import densenet_cifar
from models_kuangliu.vgg import *
from resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet110_cifar

from utils import *
from madry_model import WideResNet
from tqdm import tqdm, trange

import logging
import os
import sys

from torch.utils.data import TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly

from model_zoo_new import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# In[3]:

train_bs=128
## Changed this to 32 because DenseNet121 blows up memory
test_bs=32

test_acc_list = []
test_epoch_list = {}

for exp_ind in range(31,32):

    print("Experiement {0}: {1}".format(exp_ind, exp_name_list[exp_ind]))

    model = model_list[exp_ind].cuda()
    model = torch.nn.DataParallel(model)
    resume = loc_list[exp_ind]
    load_model = switcher.get(exp_ind, "nothing")
    load_model(model, resume)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print("Finished loading {0}".format(resume))

    transform_train, transform_test = return_transforms(data_type_list[exp_ind])

    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    test_acc = test(model, test_loader)
    test_acc_list.append(test_acc)
    
    if exp_ind in models_in_training["exp_ids"]:
        
        print("Start checking the results in all epochs!!!")
        
        epoch_list = return_epoch_list(exp_ind)
        
        test_acc_list_epochs = []
        
        for epoch_num in epoch_list:
            
            print("Experiement {0}: epoch-{1}".format(exp_ind, epoch_num))
            
            loc_epoch(exp_ind, epoch_num)
            
            model = model_list[exp_ind].cuda()
            model = torch.nn.DataParallel(model)
            resume = loc_epoch(exp_ind, epoch_num)
            load_model = switcher.get(exp_ind, "nothing")
            load_model(model, resume)
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
            print("Finished loading {0}".format(resume))

            test_acc = test(model, test_loader)
            test_acc_list_epochs.append(test_acc)
        
        test_epoch_list[exp_ind] = test_acc_list_epochs
        
    
margin_results = {"test_accs": test_acc_list, "test_epoch_accs": test_epoch_list}
    
import pickle
f = open("test_accs_model_zoo.pkl","wb")
pickle.dump(margin_results, f)
f.close()
