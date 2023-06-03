import os
import cv2
import numpy as np
import OpenEXR
import Imath
import torch

def get_num_image_channels(image_data: np.ndarray) -> int:
    """Get the number of channels in an image.

    Args:
        image_data (np.ndarray): The image data.

    Returns:
        int: The number of image channels.

    """
    # Determine the number of channels based on the dimensions of the image data
    num_channels = image_data.shape[-1] if image_data.ndim == 3 else 1

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

    # Calculate the number of channels in the image
    num_channels = len(channels)

    # Create an empty NumPy array to store the image data
    image_data = np.zeros((height, width, num_channels), dtype=np.float32)

    # Read each channel and populate the image data array
    for i, channel_name in enumerate(channels.keys()):
        # Retrieve the pixel values for the channel
        pixels = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
        pixels = np.frombuffer(pixels, dtype=np.float32)

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
    main()
