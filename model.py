# model.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class DiffusionModel(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(DiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x, noise_scale):
        x = x + noise_scale * torch.randn_like(x)  # Apply noise
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def reverse(self, x, steps=100):
        for _ in range(steps):
            x = self.forward(x, noise_scale=1.0 / steps)  # Gradually denoise
        return x
    