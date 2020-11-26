from __future__ import print_function
import torch
import argparse
from models.resnet_cifar import resnet56_cifar
from torch.utils.data import TensorDataset
import plotly
from attack_functions import *
from utils import *
from utils3d import *

parser = argparse.ArgumentParser(description='Visualize 3D models')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default = 128, 
                    help='training bs')
parser.add_argument('--test-batch-size', type=int, default = 128, 
                    help='testing bs')
parser.add_argument('--file-prefix', type=str, default = "new_plot", 
                    help='stored file name')
parser.add_argument('--resume', type=str, default ="./checkpoint/resnet56_cifar10.ckpt", 
                    help='stored model name')

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
            
lens = [[-20, 20],[-20, 20],[-20, 20]]

model = resnet56_cifar().cuda()
model = torch.nn.DataParallel(model)
ckpt_path = args.resume
model.load_state_dict(torch.load(ckpt_path))

print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print("Finished loading {0}".format(ckpt_path))

train_loader, test_loader = getData(name="cifar10", train_bs=args.batch_size, test_bs=args.test_batch_size)

test(model, test_loader)

# Here, the eps is large because we have normalized the data

PGD_attack = PGD_l2(model = model, eps = 10.0, iters = 20, alpha = 2.0)

file_path = f"./{args.file_prefix}.html"

print("Plot the visualization and save it to {0}".format(file_path))
fig = run_many(PGD_attack, 
             train_loader, 
             model, 
             subplot_grid = [3,3], 
             num_adv_directions = 1,
             lens = lens,
             resolution = "medium",
             height = 800,
             width = 800,
             show_figure = False,
             save_figure = False,
             file_path = file_path,
             title = "Visualization of ResNet56",
             )
plotly.offline.plot(fig, filename=file_path)
print("The visualization is done. Please open ./new_plot.html. The file is large and should be about 9MB. Google chrome is recommended.")