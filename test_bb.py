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
from tqdm import tqdm, trange
from models.resnet import ResNet18
from models.resnet_cifar import resnet110_cifar
from models.resnet_CIFAR100 import ResNet18 as ResNet18_CIFAR100
from models.resnet_cifar_CIFAR100 import resnet110_cifar as resnet110_cifar100

import logging
import os

from torch.utils.data import TensorDataset
import pickle

parser = argparse.ArgumentParser(description='Measure pgd error')
parser.add_argument('--name', type=str, default = "cifar10", help='dataset')
parser.add_argument('--noise-type', type=str, default = "Noisy", help='type of noise augmentation')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.0031, type=float,
                    help='perturb step size')
parser.add_argument('--file-prefix', type=str, default = "ResNet18_mixup", help='stored file name')
parser.add_argument('--resume', type=str, default = "./checkpoint/ResNet18_mixup_cifar10_type_Noisy.ckpt", help='stored model name')
parser.add_argument('--batch-size', type=int, default = 64, help='training bs')
parser.add_argument('--test-batch-size', type=int, default = 100, help='testing bs')

args = parser.parse_args()

    
def test(model, test_loader, resize=False, if_adv_distillation_model = False):
    print('Testing')
    model.eval()
    
    if resize:
        fake_input = torch.zeros(200, 3, 32, 32).cuda()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        if resize:
            data_new=F.upsample(data, size=(8,8), mode='bilinear')
            # print(data_new.size())
            fake_input[:, :, 12:20, 12:20] = data_new
            # print(fake_input[:, 10:22, 10:22].size())
            output = model(fake_input)
        
        else:
            output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.long().data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num
    

def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  normalization_factor,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size
                  ):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    x_adv = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    epsilon = epsilon * normalization_factor
    step_size = step_size * normalization_factor
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_adv = nn.CrossEntropyLoss()(F.log_softmax(model_source(x_adv), dim=1), y)
        grad = torch.autograd.grad(loss_adv, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)

    err_pgd = (model_target(x_adv).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y, normalization_factor=5.0)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)
    return robust_err_total    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

softmax1 = nn.Softmax()
        
print("Testing Black Box: {0}".format(args.resume))

if args.name == "cifar10":
    model = ResNet18()
    model_source = resnet110_cifar()
    num_classes = 10
    data_index = 10
elif args.name == "cifar100":
    num_classes = 100
    data_index = 100
    model = ResNet18_CIFAR100(num_classes)
    model_source = resnet110_cifar100(num_classes=num_classes)
else:
    raise NameError('The given dataset name is not implemented.')

if args.noise_type == "None":
    Noisy_mixup = False
elif args.noise_type == "Noisy":
    Noisy_mixup = True
else:
    raise NameError('The given noise type is not implemented.')
    
if Noisy_mixup:
    model.linear = nn.Linear(in_features=512, out_features=num_classes+1)

model_source = model_source.cuda()
model_source = torch.nn.DataParallel(model_source)    

if args.name == "cifar10":
    source_resume = "./checkpoint/resnet110_cifar10.ckpt"
elif args.name == "cifar100":
    source_resume = "./checkpoint/resnet110_cifar100.ckpt"
model_source.load_state_dict(torch.load(f"{source_resume}"))

model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load(f"{args.resume}")['net'])

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print("Finished loading {0}".format(args.resume))

train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)

test_acc = eval_adv_test_blackbox(model, model_source, device, test_loader)

test_acc_results = {"results": test_acc}

f = open("./results/test_bb/{0}.pkl".format(args.file_prefix),"wb")
pickle.dump(test_acc_results, f)
f.close()