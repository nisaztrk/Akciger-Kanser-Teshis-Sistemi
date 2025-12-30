import torch
import torch.nn as nn
from torchvision import models

class ChestCancerModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ChestCancerModel, self).__init__()
        # Notebook'ta hangi mimariyi kullanıldıysa o olmalı yani ResNet18
        self.network = models.resnet18(weights=None) 
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)