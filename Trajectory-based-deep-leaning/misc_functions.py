#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Misc helper functions """

import numpy as np
import subprocess
import torch
import torchvision.transforms as transforms
from config import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_heatmap(data, output_file=None, dpi=600, vertical_lines=None, horizontal_lines=None, v_labels=None, h_labels=None, offset=0):
    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap='gnuplot2_r', interpolation='nearest', vmin=0.3, vmax=0.5)
    cbar = plt.colorbar(img)
    #PuBuGn, RdPu, hot_r, gnuplot2_r, bwr
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=12, width=2)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.set_position([0.81,0.11,0.05,0.77])

    # Update ticks and labels
    nb_residues = data.shape[0]
    start_tick = (offset // 20) * 20
    if start_tick < offset:
        start_tick += 20

    x_tick_labels = np.arange(start_tick, nb_residues+offset, 20)
    y_tick_labels = np.arange(start_tick, nb_residues+offset, 20)

    x_positions = np.arange(start_tick - offset, nb_residues, 20)
    y_positions = np.arange(start_tick - offset, nb_residues, 20)

    ax.set_xticks(x_positions)
    ax.set_yticks(y_positions)
    ax.set_xticklabels(x_tick_labels, fontweight='bold', fontsize=12)
    ax.set_yticklabels(y_tick_labels, fontweight='bold', fontsize=12)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)

    ax.set_xlabel('Residue Index', fontweight='bold', fontsize=14)
    ax.set_ylabel('Residue Index', fontweight='bold', fontsize=14)

    # Draw vertical dashed lines
    if vertical_lines is not None and v_labels is not None:
        for i, x in enumerate(vertical_lines):
            if i == len(vertical_lines)-1:
                start_pos = 0
            else:
                start_pos = (vertical_lines[i+1] - x)/2
            ax.axvline(x, linestyle='--', color='gray', linewidth=1)
            ax.text(x+start_pos, -1.5, v_labels[i], horizontalalignment='center', fontweight='bold', fontsize=10, rotation=34)

    # Draw horizontal dashed lines
    if horizontal_lines is not None and h_labels is not None:
        for i, y in enumerate(horizontal_lines):
            if i == len(horizontal_lines)-1:
                start_pos = 0
            else:
                start_pos = (horizontal_lines[i+1] - y)/2
            ax.axhline(y, linestyle='--', color='gray', linewidth=1)
            ax.text(nb_residues -0.5, y+start_pos, h_labels[i], verticalalignment='center', fontweight='bold', fontsize=9, rotation=40)

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()


class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None

def save_saliency_map(image, saliency_map, filename, saliency_img, vertical_lines=None, horizontal_lines=None, v_labels=None, h_labels=None, offset=0):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    width, height = nb_residues, nb_residues

    print(f"saliency map shape: {saliency_map.shape}")
    saliency_map = saliency_map.reshape(width, height)
    print(f"saliency map shape: {saliency_map.shape}")
    if saliency_map.shape[0] != nb_residues or saliency_map.shape[1] != nb_residues:
        print("error: saliency map shape is not (1, nb_residues, nb_residues)")

    contacts = np.argwhere((saliency_map >= thres))
    contacts = contacts + [1, 1]
    print(filename)
    print(*contacts)

    plot_heatmap(saliency_map, output_file=saliency_img, dpi=600, vertical_lines=vertical_lines, horizontal_lines=horizontal_lines, v_labels=v_labels, h_labels=h_labels, offset=offset)
