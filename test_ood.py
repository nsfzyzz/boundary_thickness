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
from models.resnet_CIFAR100 import ResNet18 as ResNet18_CIFAR100

import logging
import os

from torch.utils.data import TensorDataset
import pickle

parser = argparse.ArgumentParser(description='Measure ood error')
parser.add_argument('--name', type=str, default = "cifar10", help='dataset')
parser.add_argument('--noise-type', type=str, default = "Noisy", help='type of noise augmentation')
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
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

softmax1 = nn.Softmax()
        
print("Testing OOD: {0}".format(args.resume))

if args.name == "cifar10":
    model = ResNet18()
    num_classes = 10
    data_index = 10
elif args.name == "cifar100":
    num_classes = 100
    data_index = 100
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

model = torch.nn.DataParallel(model)    
model = model.cuda()
model.load_state_dict(torch.load(f"{args.resume}")['net'])

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print("Finished loading {0}".format(args.resume))

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

if args.name == 'cifar10':

    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

elif args.name == 'cifar100':

    testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'fog', 'frost',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


# out-of-distribution test dataset
test_acc_c_list = []
for c_type in CORRUPTIONS:
    
    print("Testing corruption type {0}".format(c_type))
    
    X_out = np.load('./augmix/data/cifar/CIFAR-{0}-C/{1}.npy'.format(data_index, c_type))
    Y_out = np.load('./augmix/data/cifar/CIFAR-{0}-C/labels.npy'.format(data_index))
    Y_list = []
    for i in range(50000):
        Y_list.append(Y_out[i])
    testset.data = X_out
    testset.targets = Y_list
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    
    test_acc_c = test(model, test_loader)
    test_acc_c_list.append(test_acc_c)

test_acc_results = {"results": test_acc_c_list}

f = open("./results/test_ood/{0}.pkl".format(args.file_prefix),"wb")
pickle.dump(test_acc_results, f)
f.close()