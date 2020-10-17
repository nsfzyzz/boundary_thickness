import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
import argparse

from utils import *
from models.resnet import ResNet18
from models.resnet_CIFAR100 import ResNet18 as ResNet18_CIFAR100
from tqdm import tqdm, trange

import torchvision.utils

import logging
import os

from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser(description='Training noisy mixup')
parser.add_argument('--name', type=str, default = "cifar10", help='dataset')
parser.add_argument('--alpha', type=float, default = 1.0, help='dataset')
parser.add_argument('--noise-type', type=str, default = "Noisy", help='type of noise augmentation')
parser.add_argument('--num-epochs', type=int, default = 200, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default = 64, help='training bs')
parser.add_argument('--lr-max', type=float, default = 0.01, help='learning rate')
parser.add_argument('--test-batch-size', type=int, default = 200, help='testing bs')
parser.add_argument('--file-prefix', type=str, default = "ResNet18_noisy_mixup", help='stored file name')

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

# Training settings
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

if args.name == "cifar10":
    model = ResNet18()
    num_classes = 10
elif args.name == "cifar100":
    num_classes = 100
    model = ResNet18_CIFAR100(num_classes)
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
    
model = model.to(device)

model = torch.nn.DataParallel(model)
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

## Mixup training

criterion = nn.CrossEntropyLoss()

# Note that here, according to the original mixup repo, the weight_decay should be set as 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay = 1e-4)

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Loss_list = []
best_acc = 0.0

use_cuda = torch.cuda.is_available()
total_step = len(train_loader)

for epoch in range(args.num_epochs):

    model.train()

    if epoch > args.num_epochs * 0.75:
        lr = args.lr_max * 0.01
    elif epoch > args.num_epochs * 0.5:
        lr = args.lr_max * 0.1
    else:
        lr = args.lr_max

    update_lr(optimizer, lr)

    for i, (inputs, targets) in enumerate(train_loader):
        
        if Noisy_mixup:
            # generate random inputs
            inputs_random = torch.rand_like(inputs)
            # normalize
            inputs_random[:, 0, :, :] = (inputs_random[:, 0, :, :] - 0.4914) / 0.2023
            inputs_random[:, 1, :, :] = (inputs_random[:, 1, :, :] - 0.4822) / 0.1994
            inputs_random[:, 2, :, :] = (inputs_random[:, 2, :, :] - 0.4465) / 0.2010

            # noise is the 11-th class
            targets_random = torch.ones_like(targets) * 10.0

            inputs = torch.cat([inputs, inputs_random])
            targets = torch.cat([targets, targets_random])

        inputs = inputs.to(device)
        targets = targets.to(device)
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item()))
            Loss_list.append(loss.item())

    if (epoch + 1) % 20 == 0 or epoch <= 5:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, loss.item()))

        acc = test(model, test_loader)

        if acc > 0:
            print('Saving model ' + '..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{0}_epoch_{1}.ckpt'.format(args.file_prefix, epoch))

