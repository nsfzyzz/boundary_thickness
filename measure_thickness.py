from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models.resnet import ResNet18, ResNet50
from models.resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet110_cifar
from models.densenet import densenet_cifar
from models.vgg import *
from attack_functions import *
from utils import *
import pickle
import os


parser = argparse.ArgumentParser(description='Measure boundary thickness')
parser.add_argument('--name', type=str, default = "cifar10", 
                    help='dataset')
parser.add_argument('--noise-type', type=str, default = "Noisy", 
                    help='type of augmentation ("Noisy" means noisy mixup, "None" means ordinary mixup)')
parser.add_argument('--ckpt', type=str, default = "./checkpoint/ResNet18_mixup_cifar10_type_Noisy.ckpt", 
                    help='stored model name')
parser.add_argument('--eps', type=float, default=1.0, 
                    help='Attack size')
parser.add_argument('--alpha', type=float, default=0, 
                    help='output lower bound')
parser.add_argument('--beta', type=float, default=0.75, 
                    help='output upper bound')
parser.add_argument('--iters', type=int, default=20, 
                    help='The number of attack iterations')
parser.add_argument('--step-size', type=float, default=0.2, 
                    help='The attack step size')
parser.add_argument('--num-measurements', type=int, default=10, 
                    help='The number of thickness measurements/32')
parser.add_argument('--reproduce-model-zoo', default=False, action='store_true',
                    help='use the file to reproduce the results of boundary thickness on multiple models')
parser.add_argument('--exp-ind', type=int, default = 0, 
                    help='Model index when reproducing the results')
parser.add_argument('--class-pair', dest='class_pair', default=True, action='store_true',
                    help='calculate the thickness on pairs of classes')
parser.add_argument('--batch-size', type=int, default = 32, 
                    help='training bs')
parser.add_argument('--test-batch-size', type=int, default = 32, 
                    help='testing bs')
parser.add_argument('--arch', type=str, default='ResNet18',
                    help='choose the archtecure')
parser.add_argument('--num-points', type=int, default = 128, 
                    help='Number of points sampled on the segment to measure thickness.')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed (default: 1)')
parser.add_argument('--file-prefix', type=str, default = "result", 
                    help='stored file name')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def calculate_thickness(train_loader, model, PGD_attack, num_classes, args):
    
    softmax1 = nn.Softmax()
    
    temp_hist = []

    for i, (images, labels) in enumerate(train_loader):

        if i >= args.num_measurements:
            break

        print("Measuring batch {0}".format(i))

        model.eval()
        images, labels = images.cuda(), labels.cuda()

        output = model(images)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        labels_change = torch.randint(1, num_classes, (labels.shape[0],)).cuda()
        wrong_labels = torch.remainder(labels_change + labels, num_classes)
        adv_images = PGD_attack.__call__(images, wrong_labels)

        for data_ind in range(labels.shape[0]):

            x1, x2 = images[data_ind], adv_images[data_ind]

            ## We use l2 norm to measure distance
            dist = torch.norm(x1 - x2, p=2)

            new_batch = []

            ## Sample some points from each segment
            ## This number can be changed to get better precision

            for lmbd in np.linspace(0, 1.0, num=args.num_points):
                new_batch.append(x1 * lmbd + x2 * (1 - lmbd))
            new_batch = torch.stack(new_batch)

            model.eval()

            y_new_batch = softmax1(model(new_batch))

            if not args.class_pair:
                y_new_batch = y_new_batch[:, pred[data_ind]].detach().cpu().numpy().flatten()
            else:
                y_original_class = y_new_batch[:, pred[data_ind]].squeeze()
                y_target_class = y_new_batch[:, wrong_labels[data_ind]]

                y_new_batch = y_original_class - y_target_class
                y_new_batch = y_new_batch.detach().cpu().numpy().flatten()

            boundary_thickness = np.logical_and((args.beta > y_new_batch), (args.alpha < y_new_batch))

            boundary_thickness = dist.item() * np.sum(boundary_thickness) / args.num_points

            temp_hist.append(boundary_thickness)
    
    return temp_hist


if args.name == "cifar10":
    num_classes = 10
elif args.name == "cifar100":
    num_classes = 100
else:
    raise NameError('The given dataset name is not implemented.')
    
train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)

normalization_factor = 5.0

# The following lists all the models that we have used in our paper

if args.reproduce_model_zoo:
    
    from model_zoo import *

    print("Experiement {0}: {1}".format(args.exp_ind, exp_name_list[args.exp_ind]))

    print("Start checking the results in all epochs.")

    epoch_list = np.arange(10, 200, 10).tolist()

    hist_list_epochs = {}

    for epoch_num in epoch_list:

        print("Experiement {0}: epoch-{1}".format(args.exp_ind, epoch_num))

        model = model_list[args.exp_ind].cuda()
        model = torch.nn.DataParallel(model)
        ckpt = loc_epoch_list[args.exp_ind].format(epoch_num) 
        model.load_state_dict(torch.load(f"{ckpt}"))
        
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print("Finished loading {0}".format(ckpt))
        
        test(model, test_loader)
        
        PGD_attack = PGD_l2(model=model, eps=args.eps * normalization_factor, iters=args.iters,
                        alpha=args.step_size * normalization_factor)

        temp_hist = calculate_thickness(train_loader, model, PGD_attack, num_classes, args)
        
        hist_list_epochs[epoch_num] = temp_hist
    
else:    
    model_list = {
        'resnet20': resnet20_cifar(),
        'resnet32': resnet32_cifar(),
        'resnet44': resnet44_cifar(),
        'resnet56': resnet56_cifar(),
        'resnet110': resnet110_cifar(),
        'ResNet18': ResNet18(),
        'ResNet50': ResNet50(),
        'DenseNet': densenet_cifar(),
        'VGG13': VGG('VGG13'),
        'VGG19': VGG('VGG19')
    }

    model = model_list[args.arch]

    if args.noise_type == "None":
        Noisy_mixup = False
    elif args.noise_type == "Noisy":
        Noisy_mixup = True

    if args.arch=="ResNet18" and Noisy_mixup:
        model.linear = nn.Linear(in_features=512, out_features=num_classes+1, bias=True)

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(f"{args.ckpt}")
    if 'net' in checkpoint.keys():
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)

    print("Measuring boundary thickness")

    PGD_attack = PGD_l2(model=model, eps=args.eps * normalization_factor, iters=args.iters,
                        alpha=args.step_size * normalization_factor)

    temp_hist = calculate_thickness(train_loader, model, PGD_attack, num_classes, args)

    print("The boundary thickness is {0}".format(np.mean(temp_hist)))


if args.reproduce_model_zoo:
    boundary_thickness_results = {"exp_ind": args.exp_ind, "name": exp_name_list[args.exp_ind], 
                        "epoch_list":epoch_list, "results":hist_list_epochs}
    
    if not os.path.exists('./results/thickness_model_zoo'):
        os.makedirs('./results/thickness_model_zoo')
        
    f = open("./results/thickness_model_zoo/{0}_ind_{1}.pkl".format(args.file_prefix, args.exp_ind),"wb")

    pickle.dump(boundary_thickness_results, f)
    f.close()
    
else:
    boundary_thickness_results = {"results": np.mean(temp_hist)}
    
    
    f = open("./results/measure_thickness/{0}.pkl".format(args.file_prefix),"wb")
    pickle.dump(boundary_thickness_results, f)
    f.close()