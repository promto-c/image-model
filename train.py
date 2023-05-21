# train.py
import torch
import torch.optim as optim
from model import DiffusionModel
from dataset import NoisyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def train_diffusion_model(model, dataloader, epochs, device):
    model.train()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            noisy_images, clean_images = data
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            optimizer.zero_grad()

            denoised_images = model.reverse(noisy_images, steps=50)
            loss = criterion(denoised_images, clean_images)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to smaller dimension for faster training
        transforms.ToTensor(),
    ])

    dataset = NoisyDataset(root_dir='images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    model = DiffusionModel(input_channels=3, hidden_channels=64)
    train_diffusion_model(model, dataloader, epochs=10, device=device)

    # Save trained model
    torch.save(model.state_dict(), 'diffusion_model.pt')
