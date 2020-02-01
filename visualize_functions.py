from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm, trange

from torch.utils.data import TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def visualize2D(vis_net, x, y, dir1, dir2, len1 = 1, len2 = 1):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize the two directions
    
    print('Take two orthogonal directions')
    dir1 = dir1/torch.norm(dir1, p = float('2'))
    dir2 = dir2/torch.norm(dir2, p = float('2'))
    check_inner_product = torch.abs(torch.dot(dir1.view(-1), dir2.view(-1)))<0.1
    assert check_inner_product, "The inner product is too large"
    
    # Generate the visualization and data grid
    xx, yy = np.meshgrid(np.linspace(-len1, len1, 201), np.linspace(-len2, len2, 201))
    t = np.c_[xx.ravel(), yy.ravel()]
    vis_grid = torch.from_numpy(t).float().to(device)
    dirs_mat = torch.cat([dir1.reshape(1, -1), dir2.reshape(1, -1)]).to(device)
    x_grid = torch.mm(vis_grid, dirs_mat).reshape(len(vis_grid), 3, 32, 32).to('cpu') + x
        
    grid_output = []
    grid_loader = torch.utils.data.DataLoader(TensorDataset(x_grid), batch_size=64, shuffle=False, num_workers=2)
    
    for grid_points in grid_loader:
        
        grid_points = grid_points[0].to(device)
        grid_ys = vis_net(grid_points)       
        _, grid_ys = torch.max(grid_ys.data, 1)
        grid_pred = (grid_ys.to('cpu') == y)
        grid_output.append(grid_pred)
    
    y_pred = torch.cat(grid_output).cpu().reshape(xx.shape)
    
    # Plot the contour and training examples
    
    plt.contourf(xx, yy, y_pred , cmap=plt.cm.Spectral)
    plt.scatter(0, 0, s = 200, c = 'yellow', marker = 'X')
    

def find_closest_dist(x, y_adv, data_loader):
    
    closest_dist = 10000
    
    # TODO: find the closest data in the own class
    
    for i, (images, labels) in enumerate(tqdm(data_loader)):
            
        other_class_ids = labels == y_adv
        other_images = images[other_class_ids]
        if len(other_images)==0:
            continue

        dists = (other_images-x).view(len(other_images), -1)
        dists = torch.norm(dists, dim = 1)
        
        if closest_dist>torch.min(dists):
            closest_dist = torch.min(dists)
    
    return closest_dist


def find_dist_hist_on_dir(x, label, adv_dir, rand_dir, data_loader, num_return_points = 20, threshold_value = 2):
    
    threshold_value = threshold_value
    
    close_points = []
    close_points_high_dimensional = []
    
    closest_distance = 1000
    
    for i, (images, labels) in enumerate(tqdm(data_loader)):
            
        certain_class_ids = labels == label
        images = images[certain_class_ids]
        if len(images)==0:
            continue

        dists = (images - x).view(len(images), -1)
        adv_dir_new = adv_dir.view(-1)
        rand_dir_new = rand_dir.view(-1)
        
        for u in dists:
            adv_dist = torch.dot(u, adv_dir_new).item()
            rand_dist = torch.dot(u, rand_dir_new).item()
            
            if abs(adv_dist)<threshold_value:
                close_points.append([adv_dist, rand_dist])
                close_points_high_dimensional.append(u)
    
    close_points.sort(key=lambda x: x[0])
    
    print("There are " + str(len(close_points)) + " points inside this range")
    
    close_points_high_dimensional = torch.stack(close_points_high_dimensional)
    
    print(close_points_high_dimensional.shape)
    
    return np.array(close_points[:num_return_points])

def find_dist_hist_on_single_dir(x, label, adv_dir, data_loader):
        
    close_points = []    
    closest_distance = 1000
    
    for i, (images, labels) in enumerate(tqdm(data_loader)):
            
        certain_class_ids = labels == label
        images = images[certain_class_ids]
        if len(images)==0:
            continue

        dists = (images - x).view(len(images), -1)
        adv_dir_new = adv_dir.view(-1)
        
        for u in dists:
            adv_dist = torch.dot(u, adv_dir_new).item()
            close_points.append(adv_dist)
        
    return np.array(close_points)

def plot_hist_once(PGD_attack, data_loader):

    for i, (images, labels) in enumerate(data_loader):
        if i < 1 :
            x = images[0]
            y = labels[0]

            labels_change = torch.randint(1, 10, (labels.shape[0],))
            wrong_labels = torch.remainder(labels_change + labels, 10)
            adv_images = PGD_attack.__call__(images, wrong_labels)
            adv_dir = adv_images[0].cpu() - x    
            y_adv = wrong_labels[0]

            adv_dir = adv_dir/torch.norm(adv_dir, p=2)

            dist_hist_correct = find_dist_hist_on_single_dir(x, y, adv_dir, data_loader)
            dist_hist_wrong = find_dist_hist_on_single_dir(x, y_adv, adv_dir, data_loader)

    plt.figure(figsize=(10,8))
    plt.hist(dist_hist_correct, normed=True, bins=30)
    plt.ylabel('Probability')
    plt.title("Histogram of the correct class")

    plt.figure(figsize=(10,8))
    plt.hist(dist_hist_wrong, normed=True, bins=30)
    plt.ylabel('Probability')
    plt.title("Histogram of the wrong class")
    
    return
    
def plot_2D_visualization(PGD_attack, data_loader, model, if_adv_direction = True, len1 = 20, len2 = 20, num_return_points = 5000, threshold_value = 20):
    
    for i, (images, labels) in enumerate(data_loader):
        if i < 1 :
            x = images[0]
            y = labels[0]

            if if_adv_direction:
                labels_change = torch.randint(1, 10, (labels.shape[0],))
                wrong_labels = torch.remainder(labels_change + labels, 10)
                adv_images = PGD_attack.__call__(images, wrong_labels)
                adv_dir = adv_images[0].cpu() - x    
                y_adv = wrong_labels[0]
            else:
                adv_dir = torch.rand(x.shape)

            adv_dir = adv_dir/torch.norm(adv_dir, p=2)
            rand_dir = torch.rand(adv_dir.shape) - 0.5
            #rand_dir = torch.rand(adv_dir.shape)
            rand_dir = rand_dir/torch.norm(rand_dir, p=2)
            rand_dir = rand_dir - torch.dot(rand_dir.view(-1), adv_dir.view(-1))*adv_dir
            rand_dir = rand_dir/torch.norm(rand_dir, p=2)

            plt.figure(figsize=(10,8))
            plt.subplot(111)
            visualize2D(model, x, y, adv_dir, rand_dir, len1 = len1, len2 = len2)
        else:
            break
        
    if if_adv_direction:
        print("In these two directions, there is one adversarial direction")
    else:
        print("In these two directions, there is no adversarial direction")

    print("The original data is marked as an yellow X\n")
    print("Computing the closest distance from other class")
    closest_dist = find_closest_dist(x, y_adv, data_loader)
    print("The closest distance of an example from the adv class is "+ str(closest_dist.item()) + "\n")

    print("Computing the histogram of projection from other class on the adversarial direction")
    close_points_adv_class = find_dist_hist_on_dir(x, y_adv, adv_dir, rand_dir, data_loader, num_return_points = num_return_points, threshold_value = threshold_value)
    close_points_own_class = find_dist_hist_on_dir(x, y, adv_dir, rand_dir, data_loader, num_return_points = num_return_points, threshold_value = threshold_value)
    
    print("The closest points in the adversarial class are marked with black dots")
    plt.scatter(close_points_adv_class[:,0], close_points_adv_class[:,1], s = 10, c = 'black', marker = 'o')
    print("The closest points in the own class are marked with white dots")
    plt.scatter(close_points_own_class[:,0], close_points_own_class[:,1], s = 10, c = 'white', marker = 'o')
    plt.xlim(-len1,len1)
    plt.ylim(-len2,len2)
    
    return

def visualize3D(vis_net, x, y, dir1, dir2, dir3, len1 = 1, len2 = 1, len3 = 1, show_figure = True, save_figure = False, file_path = './temp.html'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize the three directions
    print('Take three orthogonal directions')
    dir1 = dir1/torch.norm(dir1, p = float('2'))
    dir2 = dir2/torch.norm(dir2, p = float('2'))
    dir3 = dir3/torch.norm(dir3, p = float('2'))
    
    # Check if the three directions are orthogonal to each other
    inner_product1 = torch.abs(torch.dot(dir1.view(-1), dir2.view(-1)))
    inner_product2 = torch.abs(torch.dot(dir1.view(-1), dir3.view(-1)))
    inner_product3 = torch.abs(torch.dot(dir2.view(-1), dir3.view(-1)))
    
    check_inner_product1 = (inner_product1<0.01).item()
    check_inner_product2 = (inner_product2<0.01).item()
    check_inner_product3 = (inner_product3<0.01).item()

    assert check_inner_product1, "The three directions are not orthogonal"
    assert check_inner_product2, "The three directions are not orthogonal"
    assert check_inner_product3, "The three directions are not orthogonal"
    
    # Generate the visualization and data grid
    #lenx, leny, lenz = 51, 51, 51
    xx, yy, zz = np.mgrid[-len1:len1:50j, -len2:len2:50j, -len3:len3:50j]

    t = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    vis_grid = torch.from_numpy(t).float().to(device)
    dirs_mat = torch.cat([dir1.reshape(1, -1), dir2.reshape(1, -1), dir3.reshape(1, -1)]).to(device)
    x_grid = torch.mm(vis_grid, dirs_mat).reshape(len(vis_grid), 3, 32, 32).to('cpu') + x
        
    grid_output = []
    grid_loader = torch.utils.data.DataLoader(TensorDataset(x_grid), batch_size=64, shuffle=False, num_workers=2)
    
    vis_net.eval()
    
    softmax1 = nn.Softmax()
    
    for grid_points in tqdm(grid_loader):
        
        grid_points = grid_points[0].to(device)
        grid_ys = vis_net(grid_points)   
        grid_ys = softmax1(grid_ys)
        grid_ys = grid_ys[:,y].detach().cpu().numpy()
        grid_output.append(grid_ys)
        
        #_, grid_ys = torch.max(grid_ys.data, 1)
        #grid_pred = (grid_ys.to('cpu') == y)
        #grid_output.append(grid_pred)
    
    y_pred0 = np.concatenate(grid_output)
    #y_pred = y_pred0.reshape(xx.shape)
        
    # set the colors of each object
    #colors = np.empty(y_pred.shape, dtype=object)
    #colors[y_pred] = 'red'
    
    # and plot everything
    fig = go.Figure(data=go.Volume(
    x=xx.flatten(),
    y=yy.flatten(),
    z=zz.flatten(),
    value=y_pred0.flatten(),
    isomin=0,
    isomax=1,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
    
    if show_figure:
        fig.show()
    
    if save_figure:
        #fig.write_image(file_path, validate=True)
        plotly.offline.plot(fig, filename=file_path)
    
    return fig


def run_once(PGD_attack, data_loader, model, if_adv_direction = True, len1 = 20, len2 = 20, len3 = 20, show_figure = True, save_figure = False, file_path = './temp.html'):

    for i, (images, labels) in enumerate(data_loader):
        if i < 1 :
            x = images[0]
            y = labels[0]

            if if_adv_direction:
                labels_change = torch.randint(1, 10, (labels.shape[0],))
                wrong_labels = torch.remainder(labels_change + labels, 10)
                adv_images = PGD_attack.__call__(images, wrong_labels)
                adv_dir = adv_images[0].cpu() - x    
                y_adv = wrong_labels[0]
            else:
                adv_dir = torch.rand(x.shape)

            adv_dir = adv_dir/torch.norm(adv_dir, p=2)
            # Normalize the first direction
            rand_dir1 = torch.rand(adv_dir.shape) - 0.5
            rand_dir1 = rand_dir1/torch.norm(rand_dir1, p=2)
            rand_dir1 = rand_dir1 - torch.dot(rand_dir1.view(-1), adv_dir.view(-1))*adv_dir
            rand_dir1 = rand_dir1/torch.norm(rand_dir1, p=2)

            # Normalize the second direction
            rand_dir2 = torch.rand(adv_dir.shape) - 0.5
            rand_dir2 = rand_dir2/torch.norm(rand_dir2, p=2)
            proj1 = torch.dot(rand_dir2.view(-1), adv_dir.view(-1))
            proj2 = torch.dot(rand_dir2.view(-1), rand_dir1.view(-1))
            rand_dir2 = rand_dir2 - proj1*adv_dir - proj2*rand_dir1
            rand_dir2 = rand_dir2/torch.norm(rand_dir2, p=2)

            fig = visualize3D(model, x, y, adv_dir, rand_dir1, rand_dir2, len1 = len1, len2 = len2, len3 = len3, show_figure = show_figure, save_figure = save_figure, file_path = file_path)
            print('x axis is adversarial direction')
            print('y axis is a random direction')
            print('y axis is a random direction')
        else:
            break
            
    return fig


def Assert_three_orthogonal(dirs):
    
    dir1, dir2, dir3 = dirs[0], dirs[1], dirs[2]
    # Check if the three directions are orthogonal to each other
    inner_product1 = torch.abs(torch.dot(dir1.view(-1), dir2.view(-1)))
    inner_product2 = torch.abs(torch.dot(dir1.view(-1), dir3.view(-1)))
    inner_product3 = torch.abs(torch.dot(dir2.view(-1), dir3.view(-1)))
    
    check_inner_product1 = (inner_product1<0.01).item()
    check_inner_product2 = (inner_product2<0.01).item()
    check_inner_product3 = (inner_product3<0.01).item()

    assert check_inner_product1, "The three directions are not orthogonal"
    assert check_inner_product2, "The three directions are not orthogonal"
    assert check_inner_product3, "The three directions are not orthogonal"
    

def Compute_grid_outputs(vis_net, x, y, dirs, lens=[[-1,1],[-1,1],[-1,1]], resolution = "high"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate the visualization and data grid
    
    if resolution == "high":
        xx, yy, zz = np.mgrid[lens[0][0]:lens[0][1]:50j, lens[1][0]:lens[1][1]:50j, lens[2][0]:lens[2][1]:50j]
    elif resolution == "medium":
        xx, yy, zz = np.mgrid[lens[0][0]:lens[0][1]:20j, lens[1][0]:lens[1][1]:20j, lens[2][0]:lens[2][1]:20j]
    elif resolution == "low":
        xx, yy, zz = np.mgrid[lens[0][0]:lens[0][1]:8j, lens[1][0]:lens[1][1]:8j, lens[2][0]:lens[2][1]:8j]
    else:
        raise NameError('The resolution has to be either high, medium, or low.')

    t = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    vis_grid = torch.from_numpy(t).float().to(device)
    dirs_mat = torch.cat([dirs[0].reshape(1, -1), dirs[1].reshape(1, -1), dirs[2].reshape(1, -1)]).to(device)
    x_grid = torch.mm(vis_grid, dirs_mat).reshape(len(vis_grid), 3, 32, 32).to('cpu')
    #print(x_grid)
    x_grid = x_grid+ x
        
    grid_output = []
    grid_loader = torch.utils.data.DataLoader(TensorDataset(x_grid), batch_size=64, shuffle=False, num_workers=2)
    
    vis_net.eval()
    
    softmax1 = nn.Softmax()
    
    for grid_points in tqdm(grid_loader):
        
        grid_points = grid_points[0].to(device)
        grid_ys = vis_net(grid_points)   
        grid_ys = softmax1(grid_ys)
        grid_ys = grid_ys[:,y].detach().cpu().numpy()
        grid_output.append(grid_ys)
        
    y_pred0 = np.concatenate(grid_output)
    
    return xx.flatten(), yy.flatten(), zz.flatten(), y_pred0.flatten()

def find_specific_class(specific_class, labels):
    
    # Find the index of the first image of a specific class
    # If there is none, return -1
    
    img_ind = -1
    
    for img_ind in range(labels.shape[0]):
        if labels[img_ind] == specific_class:
            break
    
    return img_ind

def run_many(PGD_attack, 
             data_loader, 
             model, 
             subplot_grid = [2,2], 
             num_adv_directions = 1, 
             lens = [[-1,1],[-1,1],[-1,1]], 
             resolution = "high",
             height = 1000,
             width = 1000,
             show_figure = False, 
             save_figure = False, 
             file_path = './temp.html',
             specific_class = -1,
             title = "",
             if_back_to_cpu = False):
    
    # Create a figure grid
    fig = make_subplots(
    rows=subplot_grid[0], cols=subplot_grid[1],
    specs = [[{'type': 'volume'} for _ in range(subplot_grid[1])] for ind2 in range(subplot_grid[0])])
        
    #specs=[[{'type': 'volume'}, {'type': 'volume'}],
    #       [{'type': 'volume'}, {'type': 'volume'}]])
    
    num_sub_figures_plotted = 0
        
    for i, (images, labels) in enumerate(data_loader):
        
        if if_back_to_cpu:
            images = images.cpu()
            labels = labels.cpu()
        
        if num_sub_figures_plotted < subplot_grid[0]*subplot_grid[1]:
            
            if specific_class == -1:
                
                # This means that we do not need to find a specific class
                img_ind = 0
                
            else:
                img_ind = find_specific_class(specific_class, labels)
                if img_ind == -1:
                    # This means that this batch does not contain any image of this particular class
                    print("No img of label {0}! Go to the next batch.".format(specific_class))
                    # So, go to the nect batch
                    continue
            
            x = images[img_ind]
            y = labels[img_ind]
            
            dirs = [0, 0, 0]
            if num_adv_directions == 0:
                
                print("The number of adversarial directions is 0")
                
                dirs[0] = torch.rand(x.shape) - 0.5
                dirs[1] = torch.rand(x.shape) - 0.5
                dirs[2] = torch.rand(x.shape) - 0.5
            
            elif num_adv_directions == 1:
                
                print("The number of adversarial directions is 1")
                
                labels_change = torch.randint(1, 10, (labels.shape[0],))
                wrong_labels = torch.remainder(labels_change + labels, 10)
                adv_images = PGD_attack.__call__(images, wrong_labels)
                dirs[0] = adv_images[img_ind].cpu() - x    
            
                dirs[1] = torch.rand(x.shape) - 0.5
                dirs[2] = torch.rand(x.shape) - 0.5
                
            elif num_adv_directions == 3:
                
                print("The number of adversarial directions is 3")
                
                for dir_ind in range(3):
                    
                    labels_change = torch.ones(labels.shape[0]) * (dir_ind+1)
                    labels_change = labels_change.long()
                    wrong_labels = torch.remainder(labels_change + labels, 10)
                    adv_images = PGD_attack.__call__(images, wrong_labels)
                    dirs[dir_ind] = adv_images[img_ind].cpu() - x
                    
            else:
                raise NameError('The number of adversarial directions has to be either 0, 1, or 3.')
                
            # Normalize the first direction
            dirs[0] = dirs[0]/torch.norm(dirs[0], p=2)
            
            # Normalize the second direction
            dirs[1] = dirs[1]/torch.norm(dirs[1], p=2)
            dirs[1] = dirs[1] - torch.dot(dirs[1].view(-1), dirs[0].view(-1))*dirs[0]
            dirs[1] = dirs[1]/torch.norm(dirs[1], p=2)
                
            # Normalize the third direction

            dirs[2] = dirs[2]/torch.norm(dirs[2], p=2)
            proj1 = torch.dot(dirs[2].view(-1), dirs[0].view(-1))
            proj2 = torch.dot(dirs[2].view(-1), dirs[1].view(-1))
            dirs[2] = dirs[2] - proj1*dirs[0] - proj2*dirs[1]
            dirs[2] = dirs[2]/torch.norm(dirs[2], p=2)
            
            #print("The norms of the three directions are:")
            #print(torch.norm(dirs[0], p=2))
            #print(torch.norm(dirs[0], p=2))
            #print(torch.norm(dirs[0], p=2))
            
            # Check if the three directions are orthogonal
            Assert_three_orthogonal(dirs)
            
            # Compute the grid outputs
            x, y, z, value = Compute_grid_outputs(model, x, y, dirs, lens = lens, resolution = resolution)
            
            # Figure out where to put the subfigure
            row_ind = int(num_sub_figures_plotted/subplot_grid[1])
            col_ind = num_sub_figures_plotted - row_ind*subplot_grid[1]
            
            row_ind += 1
            col_ind += 1
            
            # Add a subfigure
            fig.add_trace(
                go.Volume(
                    x=x,
                    y=y,
                    z=z,
                    value=value,
                    isomin=0,
                    isomax=1,
                    opacity=0.1, # needs to be small to see through all surfaces
                    surface_count=17, # needs to be a large number for good volume rendering
                ),
                row=row_ind, col=col_ind
            )
            
            num_sub_figures_plotted += 1 
            
        else:
            break
    
    if num_adv_directions == 0:
        title_text="All three directions are random."
    elif num_adv_directions == 1:
        title_text="X direction is adversarial."
    elif num_adv_directions == 3:
        title_text="All three directions are adversarial (with different classes)."
    else:
        raise NameError('The number of adversarial directions has to be either 0, 1, or 3.')
    
    title_text += " Exp name: "
    title_text += title
    
    fig.update_layout(height=height, width=width, title_text=title_text)
    
    if show_figure:
        fig.show()
    
    if save_figure:
        plotly.offline.plot(fig, filename=file_path)
            
    return fig


def return_3D_grid(PGD_attack, 
             data_loader, 
             model, 
             num_adv_directions = 1, 
             specific_class = -1,
             if_back_to_cpu = False):
        
    for i, (images, labels) in enumerate(data_loader):
        
        if if_back_to_cpu:
            images = images.cpu()
            labels = labels.cpu()

        if specific_class == -1:

            # This means that we do not need to find a specific class
            img_ind = 0

        else:
            img_ind = find_specific_class(specific_class, labels)
            if img_ind == -1:
                # This means that this batch does not contain any image of this particular class
                print("No img of label {0}! Go to the next batch.".format(specific_class))
                # So, go to the nect batch
                continue

        x = images[img_ind]
        y = labels[img_ind]

        dirs = [0, 0, 0]
        if num_adv_directions == 0:

            print("The number of adversarial directions is 0")

            dirs[0] = torch.rand(x.shape) - 0.5
            dirs[1] = torch.rand(x.shape) - 0.5
            dirs[2] = torch.rand(x.shape) - 0.5

        elif num_adv_directions == 1:

            print("The number of adversarial directions is 1")

            labels_change = torch.randint(1, 10, (labels.shape[0],))
            wrong_labels = torch.remainder(labels_change + labels, 10)
            adv_images = PGD_attack.__call__(images, wrong_labels)
            dirs[0] = adv_images[img_ind].cpu() - x    

            dirs[1] = torch.rand(x.shape) - 0.5
            dirs[2] = torch.rand(x.shape) - 0.5

        elif num_adv_directions == 3:

            print("The number of adversarial directions is 3")

            for dir_ind in range(3):

                labels_change = torch.ones(labels.shape[0]) * (dir_ind+1)
                labels_change = labels_change.long()
                wrong_labels = torch.remainder(labels_change + labels, 10)
                adv_images = PGD_attack.__call__(images, wrong_labels)
                dirs[dir_ind] = adv_images[img_ind].cpu() - x

        else:
            raise NameError('The number of adversarial directions has to be either 0, 1, or 3.')

        # Normalize the first direction
        dirs[0] = dirs[0]/torch.norm(dirs[0], p=2)

        # Normalize the second direction
        dirs[1] = dirs[1]/torch.norm(dirs[1], p=2)
        dirs[1] = dirs[1] - torch.dot(dirs[1].view(-1), dirs[0].view(-1))*dirs[0]
        dirs[1] = dirs[1]/torch.norm(dirs[1], p=2)

        # Normalize the third direction

        dirs[2] = dirs[2]/torch.norm(dirs[2], p=2)
        proj1 = torch.dot(dirs[2].view(-1), dirs[0].view(-1))
        proj2 = torch.dot(dirs[2].view(-1), dirs[1].view(-1))
        dirs[2] = dirs[2] - proj1*dirs[0] - proj2*dirs[1]
        dirs[2] = dirs[2]/torch.norm(dirs[2], p=2)

        #print("The norms of the three directions are:")
        #print(torch.norm(dirs[0], p=2))
        #print(torch.norm(dirs[0], p=2))
        #print(torch.norm(dirs[0], p=2))

        # Check if the three directions are orthogonal
        Assert_three_orthogonal(dirs)
        
        return x, y, dirs

        
            