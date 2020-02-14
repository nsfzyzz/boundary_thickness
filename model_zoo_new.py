# This file contains all the models used in the experiment

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
              26: resnet20_cifar(),
              27: resnet20_cifar(),
              28: resnet56_cifar(),
              29: VGG("VGG11"),
              30: resnet20_cifar(),
              31: ResNet18()
             }

suffix_list = ["R18Mx(e599)","r32Trades(e80)","R50(e600)","VGG11(e900)","R50-nolink(e600)","r56(e500)","r20(e500)","R18(e100)","R18-nd(e900)","VGG19(e900)","VGG13(e900)","VGG16(e900)","R18(e683)","R18-nolink(e600)","Dcifar(e900)","r110(ep500)","A-nd(e900)","r32(e500)","r44(e500)","r20-nd(e900)","r32-nd(e900)","r44-nd(e900)","r56-nd(e900)","r110-nd(e900)","A(e500)","D121(e900)","resnet_adv_cifar_r56_to_r20","resnet_adv_cifar_r56_to_r20_e100","r56Mx(e599)","VGG11Mx(e599)","r20Mx(e599)","R18-adv(e120)"]


loc_list = {}

for i in range(32):
    loc_list[i] = "../model_zoo/{0}.pkl".format(suffix_list[i])

def return_epoch_list(exp_id):
    
    if exp_id in [0, 28, 29, 30]:
        return [5, 9, 14, 19, 24, 49, 74, 99, 149, 199, 249, 299, 399, 499, 599]
    elif exp_id == 31:
        return [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    else:
        return [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]

models_in_training = {"exp_ids": [0, 3, 5, 6, 8, 11, 12, 16, 19, 22, 24, 28, 29, 30, 31]}
    
def loc_epoch(exp_id, epoch_num):
    loc_epoch_list = {
             0: '../model_zoo/ResNet18_mixup/ckpt_{0}.pth',
             3: '../model_zoo/VGG11/ckpt_{0}.pth',
             5: '../model_zoo/resnet56_WDpatch/net_{0}.pkl',
             6: '../model_zoo/resnet20_WDpatch/net_{0}.pkl',
              8: '../model_zoo/ResNet18_noweight_decay/net_{0}.pkl',
              11: '../model_zoo/VGG16/ckpt_{0}.pth',
              12: '../model_zoo/ResNet18_weight_decay/net_{0}.pkl',
              16: '../model_zoo/alex_cifar10_exp2/net_{0}.pkl',
              19: '../model_zoo/resnet20/net_{0}.pkl',
              22: '../model_zoo/resnet56/net_{0}.pkl',
              24: '../model_zoo/alex_WDpatch/net_{0}.pkl',
              28: '../model_zoo/resnet56_mixup/resnet_cifar_mixup_large_{0}.pth',
              29: '../model_zoo/VGG11_mixup/ckpt_{0}.pth',
              30: '../model_zoo/resnet20_mixup/ckpt_{0}.pth',
              31: '../model_zoo/ResNet18_adv/ckpt_{0}.pth'
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

def load27(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load28(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load29(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return

def load30(model, resume):
    model.load_state_dict(torch.load(f"{resume}")['net'])
    return
 
def load31(model, resume):
    model.module.load_state_dict(torch.load(f"{resume}"))
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
        26: load26,
        27: load27,
        28: load28,
        29: load29,
        30: load30,
        31: load31
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

    elif data_type == "adv_model_not_normalized":

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

exp_name_list = {0: "ResNet18 Mixup",
                1: "resnet32_cifar using Trades (aided by epoch-100)",
                2: "ResNet50 with residual links",
                3: "VGG11",
                4: "ResNet50 without residual links",
                5: "resnet56_cifar with weight decay",
                6: "resnet20_cifar with weight decay",
                7: "ResNet18 with weight decay",
                8: "ResNet18 without weight decay",
                9: "VGG19",
                10: "VGG13",
                11: "VGG16",
                12: "ResNet18 with weight decay",
                13: "ResNet18 without residual links",
                14: "DenseNet cifar",
                15: "resnet110_cifar with weight decay",
                16: "AlexNet cifar without weight decay",
                17: "resnet32_cifar with weight decay",
                18: "resnet44_cifar with weight decay",
                19: "resnet20_cifar without weight decay",
                20: "resnet32_cifar without weight decay",
                21: "resnet44_cifar without weight decay",
                22: "resnet56_cifar without weight decay",
                23: "resnet110_cifar without weight decay",
                24: "AlexNet cifar with weight decay",
                25: "DenseNet 121",
                26: "(Madry experiment) resnet56 to resnet20",
                27: "(Madry experiment) resnet56 to resnet20",
                28: "resnet56_cifar with mixup training",
                29: "VGG11 with mixup training",
                30: "resnet20_cifar with mixup training",
                31: "ResNet18 adversarial training in Madry's setting"
                } 
                
short_name_list = {0: "R18 Mx",
                1: "r32 Trades",
                2: "R50",
                3: "VGG11",
                4: "R50-nolink",
                5: "r56",
                6: "r20",
                7: "R18",
                8: "R18-nd",
                9: "VGG19",
                10: "VGG13",
                11: "VGG16",
                12: "R18",
                13: "R18-nolink",
                14: "D cifar",
                15: "r110",
                16: "A-nd",
                17: "r32",
                18: "r44",
                19: "r20-nd",
                20: "r32-nd",
                21: "r44-nd",
                22: "r56-nd",
                23: "r110-nd",
                24: "A",
                25: "D121",
                26: "Mdy-r56-r20",
                27: "Mdy-r56-r20-100",
                28: "r56 Mx",
                29: "VGG11 Mx",
                30: "r20 Mx",
                31: "R18-adv (e120)"
                }

data_type_list = {0: "normal",
                 1: "adv_model_not_normalized",
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
                 26: "normal",
                 27: "normal",
                 28: "normal",
                 29: "normal",
                 30: "normal",
                 31: "normal"}
