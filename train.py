import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize
from dataset import ImageDataset
from models.unet import UNet
from utils.model_utils import save_model
from utils.image_utils import get_num_image_channels, write_image

def train_image(model: nn.Module,
          criterion: nn.modules.loss._Loss,
          optimizer: torch.optim.Optimizer,
          dataset: Dataset,
          batch_size: int = 2, 
          start_epoch: int = 1,
          num_epochs: int = 10,
          checkpoint_path: str = str(),
          checkpoint_step: int = 5,
          device: torch.device = 'cuda',
          worker=None) -> None:
    """Train the model.

    Args:
        model (nn.Module): The model to be trained.
        criterion (nn.modules.loss._Loss): The loss criterion.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataset (Dataset): The training dataset.
        batch_size (int): The batch size.
        start_epoch (int, optional): The starting epoch. Defaults to 1.
        num_epochs (int, optional): The number of epochs to train. Defaults to 10.
        checkpoint_path (str, optional): The path to save checkpoints. Defaults to ''.
        checkpoint_step (int, optional): The interval for saving checkpoints. Defaults to 5.
        device (torch.device, optional): The device to use for training. Defaults to 'cuda'.
        worker: The worker (if applicable).
    """
    # Create the directory for saving checkpoints if it does not exist
    os.makedirs(checkpoint_path, exist_ok=True)

    # Ensure checkpoint_step is not greater than num_epochs
    if checkpoint_step > num_epochs:
        checkpoint_step = num_epochs

    # Calculate the index of the last epoch
    last_epoch = num_epochs + start_epoch - 1

    # Create data loader
    data_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the epochs
    for epoch in range(num_epochs):
        epoch += start_epoch

        # Iterate over the data loader
        for i, (inputs, labels) in enumerate(data_loader):
            # Move input and label tensors to the specified device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print epoch and loss information
        print(f'Epoch: {epoch} of {last_epoch}, Loss: {loss.item()}')

        # Save model checkpoint if checkpoint_path is specified and one of the conditions is met
        if checkpoint_path and (epoch % checkpoint_step == 0
                                or epoch == last_epoch
                                or (worker and worker.is_stopped)):
            # Save model checkpoint
            save_model(model, f'{checkpoint_path}/checkpoint_{epoch}.pth')

            # Generate the preview image path
            preview_image_path = f'{checkpoint_path}/checkpoint_{epoch}.png'

            # Get a sample of input, output, and label input
            sample_idx = 0
            input_image = inputs[sample_idx]
            output_image = outputs[sample_idx]
            label_image = labels[sample_idx]

            # Concatenate the input horizontally
            preview_image = torch.cat([input_image, output_image, label_image], dim=2)

            # Write the preview image
            write_image(preview_image_path, preview_image)

            # Check if worker is available
            if worker:
                # Worker emits the proview file
                worker.saved_checkpoint.emit(preview_image_path)
            else:
                # Print message about the preview path being saved
                print(f'Preview saved: {preview_image_path}')

def main():
    # Define paths
    data_dir = './'
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    checkpoint_path = './checkpoints'
    
    # Set hyperparameters
    num_epochs = 10
    batch_size = 2
    learning_rate = 0.001

    # Create dataset
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]
    dataset = ImageDataset(image_paths, label_paths)

    # Determine the number of channels for input and label input
    num_input_channels = get_num_image_channels(dataset[0][0])
    num_label_channels = get_num_image_channels(dataset[0][1])

    # Create model
    model = UNet(in_channels=num_input_channels, out_channels=num_label_channels)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_image(model=model,
                criterion=criterion,
                optimizer=optimizer,
                dataset=dataset,
                batch_size=batch_size, 
                num_epochs=num_epochs,
                checkpoint_path=checkpoint_path,
                device='cpu',
                )

if __name__ == '__main__':
    main()
