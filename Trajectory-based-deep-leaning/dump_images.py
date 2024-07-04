#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

import torch
from torchvision import datasets, transforms
import os
# Import saliency method
from saliency.smoothgrad import SmoothGrad
from misc_functions import *
from runcnn import ConvNet
from config import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'test_img/'
batch_size = 1
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

data_transforms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load valid data and transform
valid_dataset = datasets.ImageFolder(root=dataset, transform=data_transforms)

# get sub_dir name and the label mapping
class_to_idx = valid_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Dataset loader for sample images
sample_loader = torch.utils.data.DataLoader(valid_dataset, batch_size= batch_size, shuffle=False)

model = ConvNet(nb_classes)
model.load_state_dict(torch.load('./models/best.pt'))
model = model.to(device)

# Initialize saliency methods
saliency_methods = {
'smoothgrad': SmoothGrad(model)
}

def compute_saliency_and_save():
    vertical_lines   = [8,18,29,40,59,75,87,101,105, 111, 112,117,126,127,138,144,152]
    v_labels = ['P-loop', r'$\alpha$1', 'SW1', r'$\beta$1-$\beta$2', 'SW2', 'L2', r'$\alpha$3', '', 'L3', '',r'$\beta$5', 'L4','',r'$\alpha$4', r'$\beta$6','L5','']
    horizontal_lines = [8,18,29,40,59,75,87,101,105, 111, 112,117,126,127,138,144,152]
    h_labels = ['P-loop', r'$\alpha$1', 'SW1', r'$\beta$1-$\beta$2', 'SW2', 'L2', r'$\alpha$3', '', 'L3', '',r'$\beta$5', 'L4','',r'$\alpha$4', r'$\beta$6','L5','']
    offset = 2
    for batch_idx, (data, _) in enumerate(sample_loader):
        data = data.to(device).requires_grad_()
        labels = [idx_to_class[idx.item()] for idx in _]
        # Compute saliency maps for the input data
        for s in saliency_methods:
            saliency_map = saliency_methods[s].saliency(data)
            # Save saliency maps
            for i in range(data.size(0)):
                filename = save_path + labels[0] + str( (batch_idx+1) * (i+1))
                # image = unnormalize(data[i].cpu())
                image = data[i].cpu()
                save_saliency_map(image, saliency_map[i], filename + '_' + s + '.jpg', filename + '_' + s +'.jpg', vertical_lines=vertical_lines, horizontal_lines=horizontal_lines, v_labels=v_labels, h_labels=h_labels , offset=offset)


if __name__ == "__main__":
    # Create folder to saliency maps
    save_path = PATH + 'results'
    create_folder(save_path)
    compute_saliency_and_save()
    print('Saliency maps saved.')







