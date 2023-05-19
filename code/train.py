import sys
sys.path.append('data/')
from dataset_loader import data_loaders
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from network import Net
import copy
import time
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import config
writer = SummaryWriter("logs")


def train_model(model, criterion, optimizer, num_epochs = 25):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = {"shape": 0, "texture": 0, "combined": 0}
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*10)

        for phase in ["train","val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_shape_corrects = 0
            running_texture_corrects = 0
            for i, (inputs, labels) in enumerate(data_loaders[phase]):
                shape_label, texture_label = labels
                with torch.set_grad_enabled(phase == "train"):
                    shape_pred, texture_pred = model(inputs)                    
                    shape_loss = criterion(shape_pred,shape_label)
                    texture_loss = criterion(texture_pred,texture_label)
                    loss = shape_loss + texture_loss
                    if phase == "train":
                        n = 1000
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        n = 250
                running_loss += loss.item() * len(inputs)

                # print shape_pred to check shape
                shape_pred = torch.argmax(shape_pred, dim = 1)
                texture_pred = torch.argmax(texture_pred, dim = 1)
                shape_label = torch.argmax(shape_label, dim = 1)
                texture_label = torch.argmax(texture_label, dim = 1)
                

                running_shape_corrects += torch.sum(shape_pred == shape_label).item()
                running_texture_corrects += torch.sum(texture_pred == texture_label).item()
                num_samples = (i+1)*config.batch_size
                shape_acc = running_shape_corrects/num_samples
                texture_acc = running_texture_corrects/num_samples
                epoch_acc = (shape_acc + texture_acc) / 2
                

                if phase == "train":
                    writer.add_scalar("loss/train", running_loss/num_samples, epoch * config.batch_size + i)
                    writer.add_scalar("accuracy/Training-Shape", shape_acc, epoch * config.batch_size + i)
                    writer.add_scalar("accuracy/Training-Pattern", texture_acc, epoch * config.batch_size + i)
                    writer.add_scalar("accuracy/train", epoch_acc, epoch * config.batch_size + i)

                else: 
                    writer.add_scalar("loss/validation", running_loss/num_samples, epoch * config.batch_size + i)
                    writer.add_scalar("accuracy/Validation-Shape", shape_acc, epoch * config.batch_size+ i)
                    writer.add_scalar("accuracy/Validation-Pattern", texture_acc, epoch * config.batch_size + i)
                    writer.add_scalar("accuracy/validation", epoch_acc, epoch * config.batch_size + i)

            
            shape_acc = running_shape_corrects/n
            texture_acc = running_texture_corrects/n
            epoch_acc = (shape_acc + texture_acc) / 2

            print(f"{phase} Scaled Loss: {running_loss / n}. Shape Accuracy: {shape_acc}. Texture Accuracy: {texture_acc}")
            print()

    
            if phase == "val" and epoch_acc > best_acc["combined"]:
                best_acc["shape"] = shape_acc
                best_acc["texture"] = texture_acc
                best_acc["combined"] = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            
            if phase == "val" and best_acc["combined"] == 1:
                time_elapsed = time.time() - since
                print(f"Training completed in {time_elapsed // 60}m, {time_elapsed % 60}s")
                print(f"Best val acc: {best_acc['combined']}. Shape acc: {best_acc['shape']}. Texture acc: {best_acc['texture']}.")
                print()
                model.load_state_dict(best_model_weights)
                return model

    time_elapsed = time.time() - since
    print(f"Training completed in {time_elapsed // 60}m, {time_elapsed % 60}s")
    print(f"Best val acc: {best_acc['combined']}. Shape acc: {best_acc['shape']}. Texture acc: {best_acc['texture']}.")
    print()
    model.load_state_dict(best_model_weights)

    return model

criterion = nn.CrossEntropyLoss()
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_model = train_model(model, criterion, optimizer, num_epochs = 25)
torch.save(best_model.state_dict(), 'best_model')






