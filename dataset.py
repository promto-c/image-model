import torch
from torch.utils.data import Dataset
from utils.image_utils import read_image_as_tensor

class ImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image_tensor = read_image_as_tensor(image_path)
        label_tensor = read_image_as_tensor(label_path)

        if self.transform:
            image_tensor = self.transform(image_tensor)
            label_tensor = self.transform(label_tensor)

        return image_tensor, label_tensor
