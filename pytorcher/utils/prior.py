"""
Implement some image prior models that can be used as regularizers in the training of a DNN model.
"""

import torch

def reduction_fn(x, reduction='mean'):
    if reduction == 'mean':
        return torch.mean(x)
    elif reduction == 'sum':
        return torch.sum(x)
    elif reduction == 'none':
        return x
    else:
        raise ValueError(f"Invalid reduction method: {reduction}. Choose from 'mean', 'sum', or 'none'.")

def total_variation(x, reduction='mean'):
    """
    Compute the total variation of an image tensor.
    Args:
        x: A tensor of shape (B, C, H, W) representing a batch of images.
        reduction: A string specifying the reduction to apply to the output. Options are 'mean', 'sum', or 'none'.
    Returns:
        The total variation of the input tensor, reduced according to the specified reduction method.
    """
    # Total variation regularization
    diffX = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    diffY = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    #
    tv = torch.sum(diffX, dim=[1,2,3]) + torch.sum(diffY, dim=[1,2,3])
    #
    return reduction_fn(tv)
    
def gibbs(x, gamma=1.0,reduction='mean'):
    """
    Compute the Gibbs prior of an image tensor from :
    https://ieeexplore.ieee.org/document/998681
    A concave prior penalizing relative differences for maximum-a-posteriori reconstruction in emission tomography
    Args:
        x: A tensor of shape (B, C, H, W) representing a batch of images.
        reduction: A string specifying the reduction to apply to the output. Options are 'mean', 'sum', or 'none'.
    Returns:
        The Gibbs prior of the input tensor, reduced according to the specified reduction method.
    """
    # Gibbs prior
    quad_diffX = torch.square(x[:, :, 1:, :] - x[:, :, :-1, :])
    quad_diffY = torch.square(x[:, :, :, 1:] - x[:, :, :, :-1])
    quad_diff = torch.sum(quad_diffX, dim=[1,2,3]) + torch.sum(quad_diffY, dim=[1,2,3])
    #
    diffX = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    diffY = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    diff = torch.sum(diffX, dim=[1,2,3]) + torch.sum(diffY, dim=[1,2,3])
    #
    sumX = x[:, :, 1:, :] + x[:, :, :-1, :]
    sumY = x[:, :, :, 1:] + x[:, :, :, :-1]
    sum = torch.sum(sumX, dim=[1,2,3]) + torch.sum(sumY, dim=[1,2,3])
    #
    gibs = quad_diff / (sum + gamma * diff + 1e-8)
    #
    return reduction_fn(gibs)