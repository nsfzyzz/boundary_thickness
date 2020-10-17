from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from models.resnet import ResNet18
from models.densenet import densenet_cifar
from models.vgg import *
from resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet110_cifar

from utils import *
from tqdm import tqdm, trange

import logging
import os
import sys

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--name', type=str, default='cifar10', metavar='N',
                    help='dataset')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='learning rate ratio')
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[100,150],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', type=str, default='',
            help='choose model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--arch', type=str, default='resnet',
            help='choose the archtecure')
parser.add_argument('--saving-folder', type=str, default='',
            help='the folder to save your model')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)

model_list = {
    'resnet20': resnet20_cifar(),
    'resnet32': resnet32_cifar(),
    'resnet44': resnet44_cifar(),
    'resnet56': resnet56_cifar(),
    'resnet110': resnet110_cifar(),
    'ResNet18': ResNet18(),
    'ResNet50': ResNet50_with_link(),
    'DenseNet': densenet_cifar(),
    'VGG11': VGG('VGG11'),
    'VGG13': VGG('VGG13'),
    'VGG16': VGG('VGG16'),
    'VGG19': VGG('VGG19')
}

model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epoch)

if args.evaluate:
    if args.resume == '':
        raise ("please choose the trained model to resume")
    model.load_state_dict(torch.load(f"{args.resume}"))
    test(model, test_loader)
    sys.exit("Finishing evaluation") 

if args.saving_folder == '':
    raise ('you must give a position and name to save your model')
if args.saving_folder[-1] != '/':
    args.saving_folder += '/'
if not os.path.isdir(args.saving_folder):
    os.makedirs(args.saving_folder)
log = f"{args.saving_folder}log.log"
logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

# saving the model at 0th epoch (before training)
# torch.save(model.state_dict(), f'{args.saving_folder}net_{0}.pkl')  
total_iter = 0
best_acc = 0.0
acc_tolerance = 0
for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    with tqdm(total=len(train_loader.dataset)) as progressbar:

        for batch_idx, (data, target) in enumerate(train_loader):
            total_iter += 1
            if data.size()[0] < args.batch_size:
                continue
            model.train()
            data, target = data.cuda(), target.cuda()

            output = model(data)

            loss = criterion(output, target)
            loss.backward(create_graph = True)
            train_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            optimizer.step()
            inner_loop = 0
            optimizer.zero_grad()

            progressbar.set_postfix(
                loss=train_loss / total_num, acc=100. * correct / total_num)

            progressbar.update(target.size(0))
            
    acc = test(model, test_loader)
    train_loss /= total_num
    logging.info(f"Training Loss of Epoch {epoch}: {train_loss}")
    logging.info(f"Testing of Epoch {epoch}: {acc}")
    scheduler.step()
    torch.save(model.state_dict(), f'{args.saving_folder}net_{epoch}.pkl')  



