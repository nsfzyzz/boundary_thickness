import os
from os import listdir
from os.path import isfile, join
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def draw_once(thickness_adv_hist, mode='processed', ylim = [0, 1]):
    fig, axs = plt.subplots(1, 5, figsize=(25,5))

    markers = {"normal": 'o', "mixup": 'x', "decay": '^'}
    short_names = {"normal": "normal", "mixup": "mixup", "decay": "no decay"}
    
    networks = ["ResNet18","ResNet50","densenet","VGG13","VGG19"]
    
    for subplot_ind, network in enumerate(networks):
        
        subplot_col = int(subplot_ind%5)
        subplot_row = int(subplot_ind/5)

        ax = axs[subplot_col]

        for exp_ind in range(60):
            
            if mode=="unprocessed":
                name = thickness_adv_hist[exp_ind]["name"]
            else:
                if "{0}.pkl".format(exp_ind) not in thickness_adv_hist.keys():
                    continue
                
                name = thickness_adv_hist["{0}.pkl".format(exp_ind)]["name"]

            if network not in name:
                continue
            
            marker = None
            for marker_key in markers.keys():
                if marker_key in name:
                    marker = markers[marker_key]
                    short_name = short_names[marker_key]
                    
            if marker == None:
                continue

            if mode=="unprocessed":
                epoch_list = thickness_adv_hist[exp_ind]["epoch_list"]
            else:
                epoch_list = thickness_adv_hist["{0}.pkl".format(exp_ind)]["epoch_list"]
                
            result = []
            for epoch_num in epoch_list:
                
                if mode=="unprocessed":
                    hist = thickness_adv_hist[exp_ind]["results"][epoch_num]
                    hist = [x.cpu().item() for x in hist]
                    result.append(np.mean(hist))
                else:
                    result.append(thickness_adv_hist["{0}.pkl".format(exp_ind)]["results"][epoch_num])

            ax.plot(epoch_list, result, label = short_name, linewidth = 2, marker=marker)

        ax.set_ylim(ylim)
        ax.set_xlabel('Train epochs', fontsize=23)
        ax.set_ylabel('Boundary thickness', fontsize=23)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(loc='best', fontsize=24)
        ax.set_title(network, fontsize=22)
    
    fig.tight_layout()
    if not os.path.exists('./visualization/'):
        os.makedirs('./visualization/')
        
    plt.savefig('./visualization/Fig2.png')

mypath = './results/thickness_model_zoo/'
allfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
results = {}

for file_name in allfiles:
    if '_ind_' in file_name:
        ind = file_name.split('_ind_')[1]
        print("Processing {0}".format(file_name))

        file = open(file_name, 'rb')
        histogram = pickle.load(file)
        for key in histogram['results'].keys():
            new_list = [x for x in histogram['results'][key]]
            histogram['results'][key] = np.mean(new_list)                
        file.close()

        results[ind] = histogram

draw_once(results, ylim = [0, 3])