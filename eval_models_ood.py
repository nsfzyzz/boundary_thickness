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
from madry_model import WideResNet
from tqdm import tqdm, trange

import logging
import os
import sys

from torch.utils.data import TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly
import pickle

from model_zoo_new import *
from models_kuangliu.resnet import ResNet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
test_bs=50

# load model
model = ResNet18()
# comment the following line if the model is trained by original mixup
model.linear = nn.Linear(in_features=512, out_features=11)
model = model.to(device)
model = torch.nn.DataParallel(model)
# define model path
model_path = './checkpoint/ResNet18_mixup_199.pth'
model.load_state_dict(torch.load(model_path)['net'])


test_acc_list = []
for exp_ind in range(1):

    print("Experiement {0}: {1}".format(exp_ind, exp_name_list[exp_ind]))

    # model = model_list[exp_ind].cuda()
    # model = torch.nn.DataParallel(model)
    # resume = loc_list[exp_ind]
    # load_model = switcher.get(exp_ind, "nothing")
    # load_model(model, resume)
    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # print("Finished loading {0}".format(resume))

    transform_train, transform_test = return_transforms(data_type_list[exp_ind])
    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'fog', 'frost',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]
    # out-of-distribution test dataset
    test_acc_c_list = []
    for c_type in CORRUPTIONS:
        X_out = np.load('/home/shared5T/CIFAR-10-C/{}.npy'.format(c_type))
        Y_out = np.load('/home/shared5T/CIFAR-10-C/labels.npy')
        Y_list = []
        for i in range(50000):
            Y_list.append(Y_out[i])
        testset.data = X_out
        testset.targets = Y_list
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

        test_acc_c = test(model, test_loader)
        test_acc_c_list.append(test_acc_c)
    test_acc_list.append(test_acc_c_list)
    
margin_results = {"test_accs": test_acc_list}
    
f = open("test_accs_model_zoo_ood.pkl","wb")
pickle.dump(margin_results, f)
f.close()
