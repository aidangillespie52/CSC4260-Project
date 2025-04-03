# cnn.py

import torch
import torch.nn as nn

# Convolutional Neural Netork
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),  # Corrected size
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)  # No rounding here!
        )

    def forward(self, x):
        x = self.conv_layers(x)
        #print("After conv layers:", x.shape)  # Debugging output
        x = x.view(x.size(0), -1)  # Flatten
        #print("After flattening:", x.shape)  # Debugging output
        x = self.fc_layers(x)
        #print("After FC layers:", x.shape)  # Debugging output
        return x