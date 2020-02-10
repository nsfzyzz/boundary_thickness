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

from resnet_cifar import resnet56_cifar

from attack_functions import *
from utils import *
from visualize_functions import *
from tqdm import tqdm, trange

import logging
import os
import sys

from torch.utils.data import TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_bs=128
test_bs=128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
            
lens = [[-20, 20],[-20, 20],[-20, 20]]

model = resnet56_cifar().cuda()
model = torch.nn.DataParallel(model)
file_path = "../checkpoint/resnet56_cifar_300.pkl"
model.load_state_dict(torch.load(file_path))

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print("Finished loading {0}".format(file_path))

train_loader, test_loader = getData(name="cifar10", train_bs=train_bs, test_bs=test_bs)


test(model, test_loader)

# Here, the eps is large because we have normalized the data

PGD_attack = PGD_l2(model = model, eps = 10.0, iters = 20, alpha = 2.0)

file_path = "./new_plot.html"
print("Plot the visualization and save it to {0}".format(file_path))
fig = run_many(PGD_attack, 
             train_loader, 
             model, 
             subplot_grid = [3,3], 
             num_adv_directions = 1,
             lens = lens,
             resolution = "medium",
             height = 800,
             width = 800,
             show_figure = False,
             save_figure = False,
             file_path = file_path,
             title = "Visualization of ResNet56",
             )
plotly.offline.plot(fig, filename=file_path)
print("The visualization is done. Please open ./new_plot.html. The file is large and should be about 9MB. Google chrome is recommended.")