from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Attacks(object):
    """
    An abstract class representing attacks.

    Arguments:
        name (string): name of the attack.
        model (nn.Module): a model to attack.

    .. note:: device("cpu" or "cuda") will be automatically determined by a given model.

    """

    def __init__(self, name, model):
        self.attack = name
        self.model = model.eval()
        self.model_name = str(model).split("(")[0]
        self.device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")

    # Whole structure of the model will be NOT displayed for pretty print.
    def __str__(self):
        info = self.__dict__.copy()
        del info['model']
        del info['attack']
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    # Save image data as torch tensor from data_loader
    # If you want to reduce the space of dataset, set 'to_unit8' as True
    # If you don't want to know about accuaracy of the model, set accuracy as False
    def save(self, file_name, data_loader, to_uint8=True, accuracy=True):
        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):

            labels_change = torch.randint(1, 10, (labels.shape[0],))
            wrong_labels = torch.remainder(labels_change + labels, 10)

            adv_images = self.__call__(images, wrong_labels)

            if accuracy:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

            if to_uint8:
                image_list.append((adv_images * 255).type(torch.uint8).cpu())
            else:
                image_list.append(adv_images.cpu())

            # label_list.append(labels)
            label_list.append(predicted)

            print('- Save Progress : %2.2f %%        ' % ((step + 1) / total_batch * 100), end='\r')

            if accuracy:
                acc = 100 * float(correct) / total
                print('\n- Accuracy of the model : %f %%' % (acc), end='')

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)

        print("Return adversarial data.")

        adv_data = torch.utils.data.TensorDataset(x, y)

        return adv_data

    # Load image data as torch dataset
    # When scale=True it automatically tansforms images to [0, 1]
    def load(self, file_name, scale=True):
        adv_images, adv_labels = torch.load(file_name)

        if scale:
            adv_data = torch.utils.data.TensorDataset(adv_images.float() / adv_images.max(), adv_labels)
        else:
            adv_data = torch.utils.data.TensorDataset(adv_images.float(), adv_labels)

        return adv_data


class PGD_l2(Attacks):

    def __init__(self, model, eps=0.3, alpha=2 / 255, iters=40):
        super(PGD_l2, self).__init__("PGD_l2", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.eps_for_division = 1e-10

    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = images.data

        for i in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward()

            gradient_norms = torch.norm(images.grad.view(len(images), -1), p=2, dim=1) + self.eps_for_division
            adv_images = images - self.alpha * images.grad / gradient_norms.view(-1, 1, 1, 1)
            perturbations = adv_images - ori_images
            perturb_norms = torch.norm(perturbations.view(len(images), -1), p=2, dim=1)
            factor = self.eps / perturb_norms
            factor = torch.min(factor, torch.ones_like(perturb_norms))
            eta = perturbations * factor.view(-1, 1, 1, 1)

            images = torch.clamp(ori_images + eta, min=torch.min(images.data), max=torch.max(images.data)).detach_()

        adv_images = images

        return adv_images


def attacker_linf(net, x_nat, y, epsilon, step_size, num_steps):
    net.eval()
    x = Variable(x_nat.data, requires_grad=True)
    for _ in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(net(x), y.long())
        grad = torch.autograd.grad(loss, [x])[0]  # calculate gradient
        x = x.detach() + step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_nat - epsilon), x_nat + epsilon)
        x = torch.clamp(x, torch.min(x_nat), torch.max(x_nat))
    net.train()
    return x


def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def test_adv(model, loader, adv_test=False, use_pseudo_label=False, epsilon=8.0 / 255, step_size=2.0 / 255,
             num_steps=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_acc = 0.0
    num = 0
    total_adv_acc = 0.0

    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)

            output = model(data)

            pred = torch.max(output, dim=1)[1]
            te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

            total_acc += te_acc
            num += output.shape[0]

            if adv_test:
                # use predicted label as target label
                with torch.enable_grad():
                    adv_data = attacker_linf(model, data, label, epsilon, step_size, num_steps)

                model.eval()
                adv_output = model(adv_data)

                adv_pred = torch.max(adv_output, dim=1)[1]
                adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                total_adv_acc += adv_acc
            else:
                total_adv_acc = -num

    return total_acc / num, total_adv_acc / num