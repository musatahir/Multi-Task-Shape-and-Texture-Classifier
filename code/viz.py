import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('data/')
from dataset_loader import data_loaders
from network import Net
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision
writer = SummaryWriter("logs")

def visualize_predictions(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    shapes = ['circle', 'square', 'triangle', 'rectangle', 'star']
    patterns = ['full', 'striped', 'checkerboard', 'dotted', 'grid']
    with torch.no_grad():
    
        for i, (images, labels) in enumerate(dataloader):
            shape_label, texture_label = labels
            shape_label, texture_label = torch.argmax(shape_label, axis = 1), torch.argmax(texture_label, axis = 1)
            shape_pred, texture_pred = model(images)
            shape_pred, texture_pred = torch.argmax(shape_pred, axis = 1), torch.argmax(texture_pred, axis = 1)

            num_rows = 2
            num_cols = 2
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))  # Adjust the figure size as desired
            fig.tight_layout()
            for j, ax in enumerate(axes.flat):
                ax.axis('off')
                ax.set_title(f"Predicted Shapes: {shapes[shape_pred[j]]}. Predicted Pattern: {patterns[texture_pred[j]]}")
                ax.imshow(np.squeeze(images[j]), cmap='gray')
            plt.subplots_adjust(top=0.9)

            plt.show()
            return

            # for j in range(images.shape[0]):
            #     if j == num_images:
            #         return
            #     ax = plt.subplot(num_images//2, 2, j+1)
            #     ax.axis('off')
            #     ax.set_title(f"Predicted Shapes: {shapes[shape_pred[j]]}")
            #     plt.imshow(np.squeeze(images[j]), cmap='gray')
            #     plt.show()
        

# Create an instance of your model
model = Net()

# Load the state dictionary of the best model
model.load_state_dict(torch.load('best_model'))
ex_data = list(data_loaders["train"])[0][0]
writer.add_graph(model, ex_data)
visualize_predictions(model, data_loaders["val"])