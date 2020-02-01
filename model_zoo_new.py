# Note that this is not the original model zoo file
# This file uses the model_zoo.py
# The models'names have been changed

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
#from attack_functions import *
from models_kuangliu.resnet_without_link import ResNet50 as ResNet50_without_link
from models_kuangliu.resnet_with_link import ResNet50 as ResNet50_with_link
from models_kuangliu.resnet import ResNet18
from models_kuangliu.resnet_without_link import ResNet18 as ResNet18_without_link
from models_kuangliu.densenet import densenet_cifar, DenseNet121
from models_kuangliu.vgg import *
from resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet110_cifar

model_list = {0:ResNet18(),
             1: resnet32_cifar(),
             2: ResNet50_with_link(),
             3: VGG('VGG11'),
             4: ResNet50_without_link(),
             5: resnet56_cifar(),
             6: resnet20_cifar(),
              7: ResNet18(),
              8: ResNet18(),
              9: VGG('VGG19'),
              10: VGG('VGG13'),
              11: VGG('VGG16'),
              12: ResNet18(),
              13: ResNet18_without_link(),
              14: densenet_cifar(),
              15: resnet110_cifar(),
              16: Alex_Net(),
              17: resnet32_cifar(),
              18: resnet44_cifar(),
              19: resnet20_cifar(),
              20: resnet32_cifar(),
              21: resnet44_cifar(),
              22: resnet56_cifar(),
              23: resnet110_cifar(),
              24: Alex_Net(),
              25: DenseNet121(),
              26: ResNet18()
             }

suffix_list = ["R18Mx(e599)","r32Trades(e80)","R50(e600)","VGG11(e900)","R50-nolink(e600)","r56(e500)","r20(e500)","R18(e100)","R18-nd(e900)","VGG19(e900)","VGG13(e900)","VGG16(e900)","R18(e683)","R18-nolink(e600)","Dcifar(e900)","r110(ep500)","A-nd(e900)","r32(e500)","r44(e500)","r20-nd(e900)","r32-nd(e900)","r44-nd(e900)","r56-nd(e900)","r110-nd(e900)","A(e500)","D121(e900)"]

loc_list = {}

for i in range(26):
    loc_list[i] = "../model_zoo/{0}.pkl".format(suffix_list[i])

def return_epoch_list(exp_id):
    
    if exp_id == 26:
        return [5, 9, 14, 19, 24, 49, 74, 99, 149, 199, 249, 299, 399, 499, 599]
    else:
        return [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]

    
def loc_epoch(exp_id, epoch_num):
    loc_epoch_list = {
             3: '../model_zoo/VGG11/ckpt_{0}.pth',
             6: '../model_zoo/resnet20_WDpatch/net_{0}.pkl',
              8: '../model_zoo/ResNet18_noweight_decay/net_{0}.pkl',
              11: '../model_zoo/VGG16/ckpt_{0}.pth',
              12: '../model_zoo/ResNet18_weight_decay/net_{0}.pkl',
              16: '../model_zoo/alex_cifar10_exp2/net_{0}.pkl',
              19: '../model_zoo/resnet20/net_{0}.pkl',
              24: '../model_zoo/alex_WDpatch/net_{0}.pkl',
              26: '../model_zoo/ResNet18_mixup/ckpt_{0}.pth'

             }
    return loc_epoch_list[exp_id].format(epoch_num)
    

def load0(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load1(model, resume):
    model.module.load_state_dict(torch.load(f"{resume}"))
    return

def load2(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load3(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load4(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load5(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load6(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load7(model, resume):
    model.module.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load8(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load9(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load10(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load11(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load12(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load13(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load14(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load15(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load16(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load17(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load18(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load19(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load20(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load21(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load22(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load23(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load24(model, resume):
    model.load_state_dict(torch.load(f"{resume}"))
    return

def load25(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load26(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

switcher = {
        0: load0,
        1: load1,
        2: load2,
        3: load3,
        4: load4,
        5: load5,
        6: load6,
        7: load7,
        8: load8,
        9: load9,
        10: load10,
        11: load11,
        12: load12,
        13: load13,
        14: load14,
        15: load15,
        16: load16,
        17: load17,
        18: load18,
        19: load19,
        20: load20,
        21: load21,
        22: load22,
        23: load23,
        24: load24,
        25: load25,
        26: load26
    }


# In[4]:


def return_transforms(data_type):

    if data_type == "adv_model_downloaded":
        transform_train = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif data_type == "adv_model_yaodong":

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
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
        
    return transform_train, transform_test

exp_name_list = {0: "ResNet18 Mixup (epoch-599)",
                1: "resnet32_cifar using Trades trained for 80 epochs (aided by epoch-100)",
                2: "ResNet50 with residual links (epoch-600)",
                3: "VGG11 (epoch-900)",
                4: "ResNet50 without residual links (epoch-600)",
                5: "resnet56_cifar with weight decay (epoch-500)",
                6: "resnet20_cifar with weight decay (epoch-500)",
                7: "ResNet18 with weight decay (100 epochs)",
                8: "ResNet18 without weight decay (900 epochs)",
                9: "VGG19 (epoch-900)",
                10: "VGG13 (epoch-900)",
                11: "VGG16 (epoch-900)",
                12: "ResNet18 with weight decay (epoch-683)",
                13: "ResNet18 without residual links (epoch-600)",
                14: "DenseNet cifar (epoch-900)",
                15: "resnet110_cifar with weight decay (epoch-500)",
                16: "AlexNet cifar (epoch-900) without weight decay",
                17: "resnet32_cifar with weight decay (epoch-500)",
                18: "resnet44_cifar with weight decay (epoch-500)",
                19: "resnet20_cifar without weight decay (900 epochs)",
                20: "resnet32_cifar without weight decay (900 epochs)",
                21: "resnet44_cifar without weight decay (900 epochs)",
                22: "resnet56_cifar without weight decay (900 epochs)",
                23: "resnet110_cifar without weight decay (900 epochs)",
                24: "AlexNet cifar with weight decay (epoch-500)",
                25: "DenseNet 121 (epoch-900)",
                26: "ResNet18 with Mixup training (epoch-600)"
                }


short_name_list = {0: "R18 Mx (e599)",
                1: "r32 Trades (e80)",
                2: "R50(e600)",
                3: "VGG11 (e900)",
                4: "R50-nolink (e600)",
                5: "r56 (e500)",
                6: "r20 (e500)",
                7: "R18 (e100)",
                8: "R18-nd (e900)",
                9: "VGG19 (e900)",
                10: "VGG13 (e900)",
                11: "VGG16 (e900)",
                12: "R18 (e683)",
                13: "R18-nolink (e600)",
                14: "D cifar (e900)",
                15: "r110 (ep500)",
                16: "A-nd e900",
                17: "r32 (e500)",
                18: "r44 (e500)",
                19: "r20-nd (e900)",
                20: "r32-nd (e900)",
                21: "r44-nd (e900)",
                22: "r56-nd (e900)",
                23: "r110-nd (e900)",
                24: "A e500",
                25: "D121 (e900)",
                26: "R18 Mx new (e599)"
                }

data_type_list = {0: "normal",
                 1: "adv_model_yaodong",
                 2: "normal",
                 3: "normal",
                 4: "normal",
                 5: "normal",
                 6: "normal",
                 7: "normal",
                 8: "normal",
                 9: "normal",
                 10: "normal",
                 11: "normal",
                 12: "normal",
                 13: "normal",
                 14: "normal",
                 15: "normal",
                 16: "normal",
                 17: "normal",
                 18: "normal",
                 19: "normal",
                 20: "normal",
                 21: "normal",
                 22: "normal",
                 23: "normal",
                 24: "normal",
                 25: "normal",
                 26: "normal"}