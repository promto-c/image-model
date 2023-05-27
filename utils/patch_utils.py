import numpy as np
import torch
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


def pad_to_fit_patches(input_image_tensor: torch.Tensor, 
                       patch_size: int, 
                       overlap_size: int,
                       padding_mode: str = 'reflect') -> torch.Tensor:
    """Pad the input image tensor to ensure proper coverage of patches for patched inference.

    Args:
        input_image_tensor (torch.Tensor): The input image tensor of shape (batch, channels, height, width).
        patch_size (int): The size of each patch.
        overlap_size (int): The overlap size between patches.
        padding_mode (str, optional): The padding mode. Defaults to 'reflect'.

    Returns:
        torch.Tensor: The padded image tensor. The padding area is the additional area added around the image to accommodate the patches.

    Diagram:
    --------

    The function pads the input image tensor to ensure proper coverage of patches for patched inference.
    The diagram below illustrates the concept:

    Proper Patches:                    
    +--------+--+-----+--+--------+
    |        |  |     |  |        |
    |        |  |     |  |        |
    |        |  |     |  |        |
    +--------+--+-----+--+Patch---+
    +--------+--+-----|-- --     -|
    |        |  |     |  :        |
    |        |  |     |           |
    |        |  |     |           |
    +--------+--+-----+--+--------+

                      |-----------| ➤ patch_size
                      |--|          ➤ overlap_size

    Padded Image:
    +---------Padded Image--------+  
    |   +-----Input-Image-----+   |
    |   |                     |   |
    |   |                     |   |
    |   |             +---Patch---+
    |   |             |       |   |
    |   |             |       |   |
    |   |             |       |   |
    |   +-------------+-------+   |
    +-----------------+-----------+

        |---------------------|     ➤ image_size
                      |-----------| ➤ patch_size
                              |---| ➤ padding_size


    The padding size is calculated to ensure that patches fit properly onto the image
    with the specified overlap_size. It takes into account the overlap between patches to ensure seamless alignment. 
    """
    # Get the size of the input image by extracting the height and width dimensions
    image_size = input_image_tensor.shape[-2:]
    padding_size_y, padding_size_x = calculate_proper_padding(image_size, patch_size, overlap_size)

    # Pad the input image tensor
    padded_image_tensor = F.pad(
        input=input_image_tensor,
        pad=(padding_size_x, padding_size_x, padding_size_y, padding_size_y),
        mode=padding_mode
    )

    return padded_image_tensor

if __name__ == '__main__':
    import doctest

    # Run doctests
    doctest.testmod()
