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

from models.resnet import ResNet18, ResNet50
from models.densenet import densenet_cifar
from models.vgg import *

models = ["ResNet18", "ResNet50", "densenet_cifar", "VGG13", "VGG19"]
functions = [ResNet18, ResNet50, densenet_cifar, VGG, VGG]

types = ["mixup", "normal", "no_decay"]
types_long = ["mixup training", "normal training", "no weight decay"]

model_list = {}

loc_list = {}

loc_epoch_list = {}

short_name_list = {}

exp_name_list = {}

ind = 0

Num_models = {"num": len(models)*len(types)}

loc_prefix = './checkpoint/'

for i in range(len(functions)):
    for j in range(len(types)):
        
        # Generate models
        
        if i in [3,4]: # get VGGs
            model_list[ind] = functions[i](models[i])
        else:
            model_list[ind] = functions[i]()
        
        # Get the location of the models
        
        loc_list[ind] = loc_prefix + types[j] + '/' + models[i] + '/net_190.pkl'
        
        loc_epoch_list[ind] = loc_prefix + types[j] + '/' + models[i] + '/net_{0}.pkl'
        
        # Get the information of the models
                
        short_name_list[ind] = models[i] + '_' + types[j]
        
        exp_name_list[ind] = models[i] + ' trained with ' + types_long[j]
                
        ind += 1
