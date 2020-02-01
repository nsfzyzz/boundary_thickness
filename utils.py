import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy.linalg as LA

class Alex_Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Alex_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.classifier = nn.Linear(256, num_classes)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.features = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(64, 192, kernel_size=5, padding=2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(192, 384, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        #)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.maxpool3(x)

        #x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def getData(name='cifar10', train_bs=128, test_bs=1000, shuffle_not=True, train_index=None):    

    if name == 'mnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_bs, shuffle=False)

    if name == 'cifar10':
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

        testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)
    if name == 'cifar10_without_dataaugmentation':
        transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    if name == 'cifar100':
        
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

        trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)
    
    if name == 'svhn':
        train_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='train', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='test', download=True,transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=test_bs, shuffle=False)

    return train_loader, test_loader

def train(epoch, model, train_loader,optimizer, criterion=nn.CrossEntropyLoss()):
    model.train()
    print('\nTraing, Epoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
                    
def test(model, test_loader, resize=False):
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
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num


def exp_lr_scheduler(epoch, optimizer, strategy=True, decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    print(strategy)

    if strategy=='normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
    elif strategy=='decay':
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay_eff
    elif strategy == 'tolerance':
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_eff
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer

def exp_lr_scheduler_layerwise(epoch, optimizer, strategy=True, decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    print(strategy)

    if strategy=='normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr2'] *= decay_eff
                param_group['lr1'] = max(param_group['lr1']*decay_eff, 1e-4)
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer


def fb_warmup(optimizer, epoch, baselr, large_ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch * (baselr * large_ratio-baselr) / 5. + baselr
    return optimizer 

def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def is_substring(substrings, longstring):

    for substring in substrings:

        if substring in longstring:
            return True

    return False

def _get_weights(model):
    
    W = [param.data for name,param in model.module.named_parameters() if (is_substring(['conv', 'fc', 'linear', 'classifier'], name) and ('weight' in name))]
    #W = [getattr(model.module, name) for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'classifier']]
    #W = [getattr(model.module, name[:-7]) for name,_ in model.module.named_parameters() if (is_substring(['con1', 'fc'], name) and ('weight' in name))]

    #weights = [w.weight.view(w.weight.size()[0], -1) for w in W]
    weights = [w.view(w.size()[0], -1) for w in W]
    
    #print(weights)

    # The original code also put bias in it. I don't think this is correct
    #matrices = [w.cat((w, W.bias), dim=-1) for w, W in zip(weights, W)]
    #return matrices
    return weights

def stats(model, loader, X_fro_norm):
    global diff, row, i, idx
    #margins = torch.Tensor()
    margins = []

    i = 0
    
    model.eval()
    
    for data, target in loader:
        if i>10:
            break
        i += 1
        #if args.cuda:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output = F.log_softmax(output, dim=1)
        #print(output)

        diff = output.data.max(1, keepdim=True)[0] - output.data

        diff = diff[diff > 0].view(diff.size()[0], diff.size()[1] - 1)
        margin = diff.min(1)[0]
        margins.append(margin)

        #margins = torch.cat((margins, margin))

    margins = torch.cat(margins)
    print("Showing the margin tensor size")
    print(margins.size())

    # S = prod(max(sigma(A)) * L_i)
    # T = sum_i ||A_i - M_i]^{2/3}_1 / || ||A_i||^{2/3}_2
    # R = T**(3/2) * S
    # Lipschitz constants: relu: 1. max_pooling lipschitz: 1. log-softmax: 1
    # TODO: calculate margin with *all* data points
    A = _get_weights(model)
    sizes = [a.cpu().data.numpy().shape for a in A]
    print("The sizes of the weight matrices are")
    print(sizes)
    
    # I changed here to the Frobenius norms, which will be correct
    # The spectral norm cannot be calculated using the method here
    #L2norms = [LA.norm(a.cpu().data.numpy(), ord=2) for a in A]
    L2norms = [LA.norm(a.cpu().data.numpy(), ord='fro') for a in A]

    L1norms = [LA.norm(a.cpu().data.numpy().flat[:], ord=1) for a in A]
    T = sum(l1**(2/3) / l2**(2/3) for l1, l2 in zip(L1norms, L2norms))
    S = np.prod(L2norms)
    R = T**(3/2) * S
    n = len(loader.dataset)
    ave_margin = np.mean(margins.cpu().data.numpy())
    
    ave_margin_normalized = ave_margin/S;

    print("The spectral norm is "+str(S))
    print("The margin is ")
    print(margins)
    
    print("The normalized margin is ")
    print(ave_margin_normalized)
    
    print("The average margin is "+str(ave_margin))
    print("The other things")

    print(margins.size(), R, X_fro_norm, n)
    margin_dist = margins / (R * X_fro_norm / n)

    #return {'margin_dist': margin_dist}
    return S, ave_margin, ave_margin_normalized

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_margin(model, loader, num_batches = 10):
    global diff, row, i, idx
    margins = []

    i = 0
    
    model.eval()
    
    for data, target in loader:
        if i>num_batches:
            break
        i += 1
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output = F.log_softmax(output, dim=1)

        diff = output.data.max(1, keepdim=True)[0] - output.data

        diff = diff[diff > 0].view(diff.size()[0], diff.size()[1] - 1)
        margin = diff.min(1)[0]
        margins.append(margin)

    margins = torch.cat(margins)
    print("Showing the margin tensor size")
    print(margins.size())
    ave_margin = np.mean(margins.cpu().data.numpy())
    min_margin = np.min(margins.cpu().data.numpy())
    print("The average margin is {0} and the min margin is {1}".format(ave_margin, min_margin))
    
    return margins, ave_margin, min_margin
