# Import PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary


# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

from collections import OrderedDict

# Check versions
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

# Check for device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class BinaryImageClassifier(nn.Module):
    def __init__(self, input_channels=3, input_size=224):
        super(BinaryImageClassifier, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size

        self.features == nn.Sequential(
            #OrderedDict[
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #block 1
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #block 2
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # block 3
        )

        map_size = input_size // 8
        flattened_size = 128 * pow(map_size, 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x, threshold = 0.5):
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities >= threshold).long()
        return predictions

