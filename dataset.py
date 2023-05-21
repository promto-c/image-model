# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class NoisyDataset(Dataset):
    def __init__(self, root_dir, transform=None, noise_factor=0.5):
        self.root_dir = root_dir
        self.transform = transform
        self.noise_factor = noise_factor
        self.image_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        # Create a noisy version of the image
        noise = torch.randn_like(image) * self.noise_factor
        noisy_image = image + noise

        return noisy_image, image  # Return as a tuple of (noisy, clean)
