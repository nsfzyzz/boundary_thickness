from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils import *
from models.resnet import ResNet18
from models.resnet_CIFAR100 import ResNet18 as ResNet18_CIFAR100
import pickle


parser = argparse.ArgumentParser(description='Measure ood robustness')
parser.add_argument('--name', type=str, default = "cifar10", 
                    help='dataset')
parser.add_argument('--noise-type', type=str, default = "Noisy", 
                    help='type of augmentation ("Noisy" means noisy mixup, "None" means ordinary mixup)')
parser.add_argument('--file-prefix', type=str, default = "result", 
                    help='stored file name')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=str, default = "./checkpoint/ResNet18_mixup_cifar10_type_Noisy.ckpt", 
                    help='stored model name')
parser.add_argument('--batch-size', type=int, default = 64, 
                    help='training bs')
parser.add_argument('--test-batch-size', type=int, default = 100, 
                    help='testing bs')

args = parser.parse_args()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

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

model = torch.nn.DataParallel(model).cuda()    
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
    
    X_out = np.load('../data/ood/cifar/CIFAR-{0}-C/{1}.npy'.format(data_index, c_type))
    Y_out = np.load('../data/ood/cifar/CIFAR-{0}-C/labels.npy'.format(data_index))
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