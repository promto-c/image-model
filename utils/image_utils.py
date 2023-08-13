import os
import cv2
import numpy as np
import OpenEXR
import Imath
import torch

from typing import Union

def get_num_image_channels(image_data: Union[np.ndarray, torch.Tensor]) -> int:
    """Get the number of channels in an image.

    Args:
        image_data (Union[np.ndarray, torch.Tensor]): The image data.

    Returns:
        int: The number of image channels.

    Raises:
        ValueError: If the image data type is unsupported.

    """
    # Determine the number of channels based on the dimensions of the image data
    if isinstance(image_data, np.ndarray):
        num_channels = image_data.shape[-1] if image_data.ndim == 3 else 1
    elif isinstance(image_data, torch.Tensor):
        num_channels = image_data.shape[1] if image_data.dim() == 4 else image_data.shape[0]
    else:
        raise ValueError("Unsupported image data type. Expected np.ndarray or torch.Tensor.")

    # Return the number of channels
    return num_channels

def convert_jpg_to_exr(input_path: str, output_path: str) -> None:
    """Convert a JPEG image to an EXR file.

    Args:
        input_path (str): The path to the input JPEG image.
        output_path (str): The path to save the output EXR file.

    """
    # Read the input JPEG image
    image_data = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Convert the BGR image to RGB
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # Normalize the image data
    image_data = image_data.astype(np.float32) / 255.0

    # Create an EXR output file
    exr_file = OpenEXR.OutputFile(output_path, OpenEXR.Header(image_data.shape[1], image_data.shape[0]))

    # Prepare the channels for writing
    channels = {"R": image_data[:, :, 0].tobytes(),
                "G": image_data[:, :, 1].tobytes(),
                "B": image_data[:, :, 2].tobytes()}

    # Write the channels to the EXR file
    exr_file.writePixels(channels)

    # Close the EXR file
    exr_file.close()

def read_exr_image(image_path: str) -> np.ndarray:
    """Read an EXR image from file and return it as a NumPy array.

    Args:
        image_path (str): The path to the EXR image file.

    Returns:
        np.ndarray: The image data as a NumPy array.

    """
    # Open the EXR file for reading
    exr_file = OpenEXR.InputFile(image_path)

    # Get the image header
    header = exr_file.header()

    # Get the data window (bounding box) of the image
    data_window = header['dataWindow']

    # Get the channels present in the image
    channels = header['channels']

    # Calculate the width and height of the image
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Determine the channel keys
    channel_keys = 'RGB' if len(channels.keys()) == 3 else channels.keys()

    # Read all channels at once
    channel_data = exr_file.channels(channel_keys, Imath.PixelType(Imath.PixelType.FLOAT))

    # Create an empty NumPy array to store the image data
    image_data = np.zeros((height, width, len(channel_keys)), dtype=np.float32)

    # Populate the image data array
    for i, data in enumerate(channel_data):
        # Retrieve the pixel values for the channel
        pixels = np.frombuffer(data, dtype=np.float32)
        # Reshape the pixel values to match the image dimensions and store them in the image data array
        image_data[:, :, i] = pixels.reshape((height, width))

    return image_data

def read_image_as_tensor(image_path: str) -> torch.Tensor:
    """Read an image from file and convert it to a PyTorch tensor.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The image data as a PyTorch tensor.

    """
    if image_path.lower().endswith('.exr'):
        # Read EXR image
        image_data = read_exr_image(image_path)
    else:
        # Read image using OpenCV
        image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Convert BGR image to RGB
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    if image_data.dtype == np.uint8:
        # Normalize uint8 image data
        image_data = image_data.astype(np.float32) / 255.0

    # Convert NumPy array to PyTorch tensor
    image_tensor = torch.from_numpy(image_data).permute(2, 0, 1)

    # Return the image tensor
    return image_tensor

def write_image(image_path: str, image_tensor: torch.Tensor) -> None:
    """Write an image to disk using OpenCV.

    Args:
        image_path (str): The path to save the image.
        image_tensor (torch.Tensor): The image data as a PyTorch tensor.
    """
    # Convert the image tensor to a NumPy array
    image_data = image_tensor.permute(1, 2, 0).detach().numpy()

    # Convert image data to the appropriate data type for writing with cv2
    if image_data.dtype == np.float32:
        image_data = (image_data * 255.0).astype(np.uint8)

    # If the image data has a single channel, convert it to 3 channels (grayscale to BGR)
    if len(image_data.shape) == 2:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
    elif image_data.shape[2] == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

    # Write the image to disk using cv2
    cv2.imwrite(image_path, image_data)

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the image path
    image_path = os.path.abspath(os.path.join(current_dir, '..', 'images', 'example_image.1001.exr'))

    # Read the image and convert it to a tensor
    image_tensor = read_image_as_tensor(image_path)

    # Display the tensor shape
    print("Image shape:", image_tensor.shape)

if __name__ == "__main__":
    # main()

    import time

    exr_image_path = 'images\example_image.1003.exr'

    t0 = time.time()

    
    image_data = read_exr_image(exr_image_path)

    # image_data = cv2.imread(ex
    # r_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(image_data.shape)


    t1 = time.time()

    print(t1-t0)
