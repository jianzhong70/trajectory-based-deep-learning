import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
from runcnn import ConvNet
from config import *


def plot_confusion_matrix(confusion_matrix, labels, save_path):
    plt.imshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=12, width=2)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold')

    # Showing the confusion values in the corresponding grids
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, "{:.4f}".format(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > (confusion_matrix.max() / 2) else "black", fontweight='bold')

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontweight='bold')   #, rotation=45
    plt.yticks(tick_marks, labels, fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold', fontsize=16)
    plt.ylabel('True Class', fontweight='bold', fontsize=16)
    # plt.title('Confusion Matrix', fontweight='bold', fontsize=16)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.tick_params(width=2)
    
    plt.grid(False)
    plt.tight_layout()

    # plt.show()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.ion()
    plt.pause(1)
    plt.close()


model = ConvNet(nb_classes)
model.load_state_dict(torch.load('./models/best.pt'))

img_width, img_height = nb_residues, nb_residues
valid_data_dir = '../valid/'

nb_valid_samples = int(0.2 * nb_systems * total_prod_steps * 0.002 / stride)
batch_size = 512

valid_transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

valid_dataset = ImageFolder(valid_data_dir, transform=valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

valid_pred = []
model.eval()
with torch.no_grad():
    for inputs, _ in valid_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        valid_pred.extend(predicted.cpu().numpy())

valid_pred = np.array(valid_pred)

cf_matrix = confusion_matrix(valid_dataset.targets, valid_pred)
print(cf_matrix)
cf_matrix = cf_matrix / cf_matrix.astype(float).sum(axis=1)
print(cf_matrix)

labels = ['SOS1 G12C', 'G12 KRAS', 'WT KRAS']
save_path = './models/conf-matrix.jpg'
# Plotting the confusion matrix to show the results of DL
plot_confusion_matrix(cf_matrix, labels, save_path)
