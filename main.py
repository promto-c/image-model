# main.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import NoisyDataset
from model import DiffusionModel
from train import train_diffusion_model
from inference import denoise_images

def main():
    # Define device and transformations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to smaller dimension for faster training
        transforms.ToTensor(),
    ])
    
    # Create a dataset and a data loader
    dataset = NoisyDataset(root_dir='images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Create model
    model = DiffusionModel(input_channels=3, hidden_channels=64)

    # Train model
    train_diffusion_model(model, dataloader, epochs=10, device=device)

    # Save trained model
    torch.save(model.state_dict(), 'diffusion_model.pt')

    # Load model for inference
    model.load_state_dict(torch.load('diffusion_model.pt'))
    model = model.to(device)

    # Use some noisy images for inference
    noisy_images, _ = next(iter(dataloader))  # Using the first batch from dataloader
    denoised_images = denoise_images(model, noisy_images, device)

    # TODO: Add code here to visualize or save denoised images if needed

if __name__ == '__main__':
    main()
