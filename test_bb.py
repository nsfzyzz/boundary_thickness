from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from models.resnet import ResNet18
from models.resnet_cifar import resnet110_cifar
from models.resnet_CIFAR100 import ResNet18 as ResNet18_CIFAR100
from models.resnet_cifar_CIFAR100 import resnet110_cifar as resnet110_cifar100
import pickle


parser = argparse.ArgumentParser(description='Measure blackbox robustness')
parser.add_argument('--name', type=str, default = "cifar10", 
                    help='dataset')
parser.add_argument('--noise-type', type=str, default = "Noisy", 
                    help='type of augmentation ("Noisy" means noisy mixup, "None" means ordinary mixup)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed (default: 1)')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.0031, type=float,
                    help='perturb step size')
parser.add_argument('--file-prefix', type=str, default = "result", 
                    help='stored file name')
parser.add_argument('--ckpt', type=str, default = "./checkpoint/ResNet18_mixup_cifar10_type_Noisy.ckpt", 
                    help='stored model name')
parser.add_argument('--batch-size', type=int, default = 64, 
                    help='training bs')
parser.add_argument('--test-batch-size', type=int, default = 100, 
                    help='testing bs')

args = parser.parse_args()

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


def eval_adv_test_blackbox(model_target, model_source, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y, normalization_factor=5.0)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)
    return robust_err_total    


# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

softmax1 = nn.Softmax()
        
print("Testing Black Box: {0}".format(args.ckpt))

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

# In the paper, we use resnet110 as the source model
if args.name == "cifar10":
    source_resume = "./checkpoint/resnet110_cifar10.ckpt"
elif args.name == "cifar100":
    source_resume = "./checkpoint/resnet110_cifar100.ckpt"
model_source.load_state_dict(torch.load(f"{source_resume}"))

model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(f"{args.ckpt}")['net'])

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print("Finished loading {0}".format(args.ckpt))

train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)

test_acc = eval_adv_test_blackbox(model, model_source, test_loader)

test_acc_results = {"results": test_acc}

f = open("./results/test_bb/{0}.pkl".format(args.file_prefix),"wb")
pickle.dump(test_acc_results, f)
f.close()