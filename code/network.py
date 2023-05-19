import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from collections import OrderedDict

# Define the final output layer as a concatenation of the shape and texture branches
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # for param in resnet.parameters():
        #     param.requires_grad = False
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.n_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.net = resnet
        self.fc_shape = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 5))]))
        self.fc_texture = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 5))]))
    
    def forward(self, x):
        x = self.net(x)
        shape_pred = self.fc_shape(x)
        texture_pred = self.fc_texture(x)
        return shape_pred, texture_pred
