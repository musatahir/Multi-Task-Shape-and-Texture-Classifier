import os
from imutils import paths
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import torch
import sys
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
import time
sys.path.append('code/')
import config



class ShapeDataset(Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        self.image_paths = gen_image_paths()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img_tokens = image_path.split("_")
        shape = img_tokens[0]
        pattern = img_tokens[1]
        img = Image.open(f"data/saved-images/{shape}/{pattern}/{image_path}")
        y_pair = gen_y_labels(image_path)
        if self.transform:
            img = self.transform(img)
        return img, y_pair

    

def gen_image_paths():
    shape_dirs = [os.path.join("data/saved-images", x) for x in os.listdir("data/saved-images")]
    image_paths = []
    for shape in shape_dirs:
        pattern_dirs = [os.path.join(shape, x) for x in os.listdir(shape)]
        for pattern in pattern_dirs:
            image_paths.extend(os.listdir(pattern))
    return image_paths


def gen_y_labels(img_path):
    shapes = ['circle', 'square', 'triangle', 'rectangle', 'star']
    patterns = ['full', 'striped', 'checkerboard', 'dotted', 'grid']
    img_tokens = img_path.split("_")
    shape = img_tokens[0]
    pattern = img_tokens[1]
    label1 = torch.zeros(len(shapes))
    label2 = torch.zeros(len(patterns))
    label1[shapes.index(shape)] = 1
    label2[patterns.index(pattern)] = 1
    return (label1, label2)


transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), 
                                transforms.RandomHorizontalFlip()])

dataset = ShapeDataset(transform = transform)
labels = []
for data in dataset:
    x, (label1, label2) = data
    label1 = torch.argmax(label1).item()
    label2 = torch.argmax(label2).item()
    labels.append((label1,label2))

splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.val_split, random_state=42)
train_idx, test_idx = next(splitter.split(dataset, labels))

train_set = [dataset[i] for i in train_idx]
test_set = [dataset[i] for i in test_idx]
train_loader = DataLoader(dataset = train_set, batch_size = config.batch_size, shuffle=True)

test_loader = DataLoader(dataset = test_set, batch_size = config.batch_size, shuffle=True)
data_loaders = {"train": train_loader, "val": test_loader}
