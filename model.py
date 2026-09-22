import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=20):
    resnet = models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet
