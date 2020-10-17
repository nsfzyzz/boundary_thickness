from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.resnet import ResNet18
from attack_functions import *
from utils import *


parser = argparse.ArgumentParser(description='Measure boundary thickness')
parser.add_argument('--eps', type=float, default=1.0, help='Attack size')
parser.add_argument('--alpha', type=float, default=0, help='output lower bound')
parser.add_argument('--beta', type=float, default=0.75, help='output upper bound')
parser.add_argument('--iters', type=int, default=20, help='The number of attack iterations')
parser.add_argument('--step-size', type=float, default=0.2, help='The attack step size')
parser.add_argument('--num-measurements', type=int, default=10, help='The number of thickness measurements/32')
parser.add_argument('--class-pair', dest='class_pair', default=True, action='store_true',
                    help='calculate the thickness on pairs of classes')

args = parser.parse_args()

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_bs = 32
test_bs = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)
testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

noise_sd = 1.0
normalization_factor = 5.0

softmax1 = nn.Softmax()

model = ResNet18()
resume = "./checkpoint/ResNet18_mixup_cifar10_type_Noisy.ckpt"
model.linear = torch.nn.Linear(in_features=512, out_features=11, bias=True)

model = model.cuda()
model = torch.nn.DataParallel(model)

checkpoint = torch.load(f"{resume}")
model.load_state_dict(checkpoint['net'])

num_lines = 0

print("Measuring boundary thickness")

PGD_attack = PGD_l2(model=model, eps=args.eps * normalization_factor, iters=args.iters,
                    alpha=args.step_size * normalization_factor)

temp_hist = []

for i, (images, labels) in enumerate(train_loader):

    if i >= args.num_measurements:
        break

    print("Measuring batch {0}".format(i))

    model.eval()
    images, labels = images.cuda(), labels.cuda()

    output = model(images)
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

    labels_change = torch.randint(1, 10, (labels.shape[0],)).cuda()
    wrong_labels = torch.remainder(labels_change + labels, 10)
    adv_images = PGD_attack.__call__(images, wrong_labels)

    for data_ind in range(labels.shape[0]):

        ## Sample 128 points from each segment

        x1, x2 = images[data_ind], adv_images[data_ind]
        dist = torch.norm(x1 - x2, p=2)

        new_batch = []

        num_points = 128
        for lmbd in np.linspace(0, 1.0, num=num_points):
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

        boundary_thickness = dist.item() * np.sum(boundary_thickness) / num_points

        temp_hist.append(boundary_thickness)

print("The boundary thickness is {0}".format(np.mean(temp_hist)))
