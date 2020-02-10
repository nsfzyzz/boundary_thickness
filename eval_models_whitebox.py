from __future__ import print_function
from torch.utils.data import TensorDataset
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
import copy
import logging
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
from model_zoo_new import *
from models_kuangliu.resnet import ResNet18

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031 * 5,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003 * 5,
                    help='perturb step size')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# In[3]:

train_bs=128
test_bs=50

test_acc_list = []

# load model
model = ResNet18()
# comment the following line if the model is trained by original mixup
model.linear = nn.Linear(in_features=512, out_features=11)
model = model.to(device)
model = torch.nn.DataParallel(model)
# define model path
model_path = './checkpoint/ResNet18_mixup_199.pth'
model.load_state_dict(torch.load(model_path)['net'])


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    x_adv = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_adv = nn.CrossEntropyLoss()(F.log_softmax(model(x_adv), dim=1), y)
        grad = torch.autograd.grad(loss_adv, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)

    err_pgd = (model(x_adv).data.max(1)[1] != y.data).float().sum()
    acc_10 = (model(x_adv).data.max(1)[1] == 10).float().sum()
    print('predicted 10th class:', acc_10)
    # raise RuntimeError
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)
    return robust_err_total


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

    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    test_acc = eval_adv_test_whitebox(model, device, test_loader)
    test_acc_list.append(test_acc)
    
margin_results = {"test_accs": test_acc_list}
    
import pickle
f = open("test_accs_model_zoo_wb.pkl","wb")
pickle.dump(margin_results, f)
f.close()
