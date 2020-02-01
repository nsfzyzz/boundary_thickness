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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

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

eps_list = [50.0/255, 40.0/255, 30.0/255, 20.0/255, 10.0/255, 8.0/255, 6.0/255, 4.0/255, 2.0/255]

def generate_PGD_attack_list(exp_ind):
    
    # This represents the collection of different attack sizes
    # 40.0/255 = 5* 8/255 represents the usual attack size
    
    PGD_attack_list = []
    
    for i in range(len(eps_list)):
        eps = eps_list[i]
        if exp_ind == 1:
            eps = eps/5
        alpha = eps/5
        
        PGD_attack_list.append([eps, alpha])
                    
    return PGD_attack_list

test_acc_list = []
robust_acc_list = []

for exp_ind in range(24):

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

    PGD_attack_list = generate_PGD_attack_list(exp_ind)
    
    robust_accs = []
    for i in range(len(PGD_attack_list)):
        
        test_acc, robust_acc = test_adv(model, test_loader, adv_test=True, epsilon = PGD_attack_list[i][0], 
                                        step_size = PGD_attack_list[i][1], num_steps = args.iters)
        
        print("The attack has eps {0}, num iters {1}, step size {2}".format(PGD_attack_list[i][0], 
                                                                            args.iters, PGD_attack_list[i][1]))
        print("The robust accuracy is {0}".format(robust_acc))
        
        robust_accs.append(robust_acc)
        
    robust_acc_list.append(robust_accs)
    
robust_acc_result = {"eps_list": eps_list,
                        "test_accs": test_acc_list,
                        "robust_accs": robust_acc_list}
    
import pickle
f = open("robust_acc_model_zoo_wb.pkl","wb")
pickle.dump(robust_acc_result, f)
f.close()
