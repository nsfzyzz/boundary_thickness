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

from attack_functions import *
from utils import *
from visualize_functions import *
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

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--iters', type=int, default = 20, help='Number of PGD attack iterations')
parser.add_argument('--eps', type=float, default = 5*8.0/255, help='PGD attack epsilon')
parser.add_argument('--alpha', type=float, default = 5*1.6/255, help='PGD attack step size')

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))
    
# In[2]:


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# In[3]:


train_bs=128
test_bs=16


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# Choose the experiments to visualize
exp_ind_list = [3, 6, 8, 11, 12, 16, 19, 24, 26]

robust_acc_result = {}
    
for exp_ind in exp_ind_list:
    
    epoch_list = return_epoch_list(exp_ind)

    print("Experiement {0}: {1}".format(exp_ind, exp_name_list[exp_ind]))

    transform_train, transform_test = return_transforms(data_type_list[exp_ind])

    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    robust_acc_list = {}
    for epoch_num in epoch_list:
        
        print("Testing robust accuracy of epoch {0}".format(epoch_num))

        model = model_list[exp_ind].cuda()
        model = torch.nn.DataParallel(model)

        # Load model
        resume = loc_epoch(exp_ind, epoch_num)
        print("Loading: {0}".format(resume))

        load_model = switcher.get(exp_ind, "nothing")
        load_model(model, resume)
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print("Finished loading {0}".format(resume))

        test(model, test_loader)
        
        test_acc, robust_acc = test_adv(model, test_loader, adv_test=True, epsilon = args.eps, 
                                    step_size = args.alpha, num_steps = args.iters)            
        
        ## Gaussian perturbation code
        
        print("The attack has eps {0}, num iters {1}, step size {2}".format(args.eps, args.iters,args.alpha))
        print("The robust accuracy is {0}".format(robust_acc))

        robust_acc_list[epoch_num] = robust_acc

    robust_acc_result[exp_ind] = {"name": exp_name_list[exp_ind], "result": robust_acc_list}
    
import pickle
f = open("robust_acc_during_training.pkl","wb")
pickle.dump(robust_acc_result, f)
f.close()
