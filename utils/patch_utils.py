import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numbers import Number
from typing import Iterable, Union

def calculate_proper_padding(image_size: Union[Iterable[Number], Number], 
                             patch_size: int, 
                             overlap_size: int) -> Union[np.ndarray, int]:
    """Calculate the proper padding size for an image to fit patches.

    The padding size is calculated to ensure that patches fit properly onto the image
    with the specified overlap_size.

    Args:
        image_size (Union[Iterable[Number], Number]):
            The dimensions of the input image can be provided as an iterable of numbers representing each dimension.
            They will be converted to a NumPy array.
            Alternatively, a single number can be provided if all dimensions are equal.
        patch_size (int): The size of each patch.
        overlap_size (int): The overlap size between patches.

    Returns:
        Union[np.ndarray, int]:
            The proper padding size as a NumPy array for any iterable, and as an integer for any single number.
            The dimensions (shape) of the returned value will match the input `image_size`.

    Example:
        Example usages of the function:

        >>> calculate_proper_padding(image_size=(720, 1024), 
        ...                          patch_size=256, 
        ...                          overlap_size=32)
        array([104,  64], dtype=int32)

        >>> calculate_proper_padding(64, 16, 4)
        0
    
    Diagram:
    --------
    The function calculates the padding size needed to fit patches of size 'patch_size' onto
    an input image of size 'image_size'. The patches are placed with a specified 'overlap_size'
    to ensure proper coverage of the image.

    The diagram below illustrates the concept:

    Padded Image:
    +--------+Padded-Image--------+
    |   +----+Input-Image+----+   |
    |   |    |  |     |  |    |   |
    |   |    |  |     |  |    |   |
    +---+----+--+-----+---Patch---+
    +---+----+--+-----|       :   +
    |   |    |  |     |           |
    |   |    |  |     |       :   |
    |   +----+--+-----+ -+-  -+   |
    +--------+--+-----+--+----+===+

        |---------------------|     ➤ image_size
                      |-----------| ➤ patch_size
                      |--|          ➤ overlap_size
                              |===| ➤ symmetric_padding_size (Returns this value)

    The padding size is calculated to ensure that patches fit properly onto the image
    with the specified overlap_size.
    """
    # Convert image_size to NumPy array if it is an iterable for mathematical evaluation
    if isinstance(image_size, Iterable):
        image_size = np.array(image_size)

    # Calculate the stride by subtracting the overlap size from the patch size
    stride = patch_size - overlap_size

    # Calculate the padding size required to fit patches onto the image
    padding_size = stride - ((image_size - overlap_size) % stride)
    
    # Set padding size to 0 on dimensions where stride completely divides the difference
    if isinstance(image_size, Iterable):
        padding_size[padding_size == stride] = 0
    elif padding_size == stride:
        padding_size = 0

    # Adjust the padding size to ensure symmetric padding if necessary
    symmetric_padding_size = (padding_size + 1) // 2

    return symmetric_padding_size

def stitch_patches(patches_tensor: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """Stitch together patches to reconstruct the original image.

    Args:
        patches_tensor (torch.Tensor): Tensor containing patches.
            Shape: (channel, num_patch_y, num_patch_x, patch_size_y, patch_size_x)
        overlap_size (int): Size of the overlap between patches.
        merge_operation (str, optional): Merge operation for blending the overlapping areas.
            Supported values: 'sum' (default), 'average', 'linear_blend'.

    Returns:
        torch.Tensor: Stitched image tensor.
            Shape: (channel, image_size_y, image_size_x)

    Merging process: In the overlap area, the values from different patches are summed together.
    Each overlapping pixel combines the values from different patches, ensuring proper blending.
    """
    # Get the shape values after permuting the dimensions
    channel, num_patch_y, num_patch_x, patch_size_y, patch_size_x = patches_tensor.shape

    # Calculate the stitched image size
    image_size_y = num_patch_y * (patch_size_y - overlap_size) + overlap_size
    image_size_x = num_patch_x * (patch_size_x - overlap_size) + overlap_size

    # Create the stitched image tensor
    stitched_image_tensor = torch.zeros((channel, image_size_y, image_size_x), dtype=patches_tensor.dtype)

    # Iterate over the patches and stitch them together
    for i in range(num_patch_x):
        for j in range(num_patch_y):
            # Calculate the starting position for the current patch
            start_y = j * (patch_size_y - overlap_size)
            start_x = i * (patch_size_x - overlap_size)

            # Calculate the ending position for the current patch
            end_y = start_y + patch_size_y
            end_x = start_x + patch_size_x

            # Merge the patch with the stitched image, considering the overlapping edges
            stitched_image_tensor[:, start_y:end_y, start_x:end_x] += patches_tensor[:, j, i]

    # Permute the dimensions to match the original shape
    return stitched_image_tensor

def patched_inference(model: nn.Module,
                      input_image_tensor: torch.Tensor,
                      patch_size: int,
                      overlap_size: int = 0) -> torch.Tensor:
    """Perform patched inference on an input image by dividing it into patches, applying the model to each patch,
    and stitching the output patches back together.

    Args:
        model (nn.Module): The model to use for inference.
        input_image_tensor (torch.Tensor): The input image tensor of shape (batch, num_channels , height, width).
        patch_size (int): The size of each patch.
        overlap_size (int, optional): The overlap size between patches. Defaults to 0.

    Returns:
        torch.Tensor: The output image tensor of shape (batch, num_channels , height, width).
    """

    # Padding
    # -------
    # Adjust the overlap size to ensure it is within the valid range
    overlap_size = min(overlap_size, patch_size - 1)

    # Get the size of the input image by extracting the height and width dimensions
    image_size = input_image_tensor.shape[-2:]
    padding_size_y, padding_size_x = calculate_proper_padding(image_size, patch_size, overlap_size)

    # Pad the input image tensor
    # Input shape: (batch, num_channels , height, width)
    # Output shape: (batch, num_channels , padded_height, padded_width)
    padded_image_tensor = F.pad(
        input=input_image_tensor,
        pad=(padding_size_x, padding_size_x, padding_size_y, padding_size_y),
        mode='reflect',
    )

    # Patching
    # --------
    # Calculate the stride for extracting patches
    stride = patch_size - overlap_size

    # Squeeze out the batch dimension
    # Input shape: (batch, num_channels , padded_height, padded_width)
    # Output shape: (num_channels , padded_height, padded_width)
    padded_image_tensor = padded_image_tensor.squeeze()

    # Unfold or patching to smaller size patches image
    # Output shape: (channel, num_patch_y, num_patch_x, patch_size_y, patch_size_x)
    patches_tensor = padded_image_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)

    # Inference
    # ---------
    # Extract original patches shape
    original_patches_shape = patches_tensor.shape
    num_channels , num_patch_y, num_patch_x, patch_size_y, patch_size_x = original_patches_shape

    # Flatten patches
    # Input shape: (channel, num_patch_y, num_patch_x, patch_size_y, patch_size_x)
    # Output shape: (channel, num_patch_y*num_patch_x, patch_size_y, patch_size_x)
    patches_tensor = patches_tensor.reshape(num_channels , num_patch_x * num_patch_y, patch_size_y, patch_size_x)

    # Rearrange dimension order
    # Output shape: (num_patch_y*num_patch_x, channel, patch_size_y, patch_size_x)
    patches_tensor = patches_tensor.permute(1, 0, 2, 3)

    # Perform inference on each patch using the model
    for index, image_tensor in enumerate(patches_tensor):
        patches_tensor[index] = model(image_tensor.unsqueeze(0))

    # Rearrange dimension order bsck to channel first
    # Output shape: (channel, num_patch_y*num_patch_x, patch_size_y, patch_size_x)
    patches_tensor = patches_tensor.permute(1, 0, 2, 3)

    # Reshape the output tensor to match the original patches shape
    # Output shape: (channel, num_patch_y, num_patch_x, patch_size_y, patch_size_x)
    patches_tensor = patches_tensor.reshape(original_patches_shape)

    # Patch Blending
    # --------------
    # Create a tensor `weight` with values ranging from 0 to 1, evenly spaced.
    weight_tensor = torch.linspace(0, 1, overlap_size, device=patches_tensor.device)

    # Blend the overlap edges horizontally
    patches_tensor[:, :-1, :, :, -overlap_size:] *= 1 - weight_tensor
    patches_tensor[:, 1:, :, :, :overlap_size] *= weight_tensor

    # Blend the overlap edges vertically
    patches_tensor[:, :, :-1, -overlap_size:] *= 1 - weight_tensor.unsqueeze(1)
    patches_tensor[:, :, 1:, :overlap_size] *= weight_tensor.unsqueeze(1)
    
    # Stitching
    # ---------
    # Stitch the output patches together to reconstruct the final output image
    output_image_tensor = stitch_patches(patches_tensor, overlap_size)

    # Crop back to original size
    output_image_tensor = output_image_tensor[
        :, padding_size_y : padding_size_y + image_size[0], padding_size_x : padding_size_x + image_size[1]
    ]

    # Return the output image tensor
    return output_image_tensor.unsqueeze(0)

if __name__ == '__main__':
    import doctest

    # Run doctests
    doctest.testmod()
