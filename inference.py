# inference.py
import torch
from model import DiffusionModel
from dataset import NoisyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def denoise_images(model, noisy_images, device):
    model.eval()
    model.to(device)
    noisy_images = noisy_images.to(device)

    with torch.no_grad():
        denoised_images = model.reverse(noisy_images, steps=50)

    return denoised_images

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel(input_channels=3, hidden_channels=64)
    model.load_state_dict(torch.load('diffusion_model.pt'))
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to smaller dimension for faster training
        transforms.ToTensor(),
    ])

    dataset = NoisyDataset(root_dir='images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    noisy_images, _ = next(iter(dataloader))
    denoised_images = denoise_images(model, noisy_images, device)

    # TODO: Add code here to visualize or save denoised images