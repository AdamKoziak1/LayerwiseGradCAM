#!/usr/bin/env python
"""flashtorch.utils

This module provides utility functions for image handling and tensor
transformation.

"""
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
from .imagenet import *
import numpy as np

def load_image(image_path):
    """Loads image as a PIL RGB image.

        Args:
            - **image_path (str) - **: A path to the image

        Returns:
            An instance of PIL.Image.Image in RGB

    """

    return Image.open(image_path).convert('RGB')


def apply_transforms(image, size=224):
    """Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of :math:`(N, C, H, W)` and
    is ready to be used as an input to neural networks.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.

    The plan is to to expand this to handle custom size/mean/std.

    Args:
        image (PIL.Image.Image or numpy array)
        size (int, optional, default=224): Desired size (width/height) of the
            output tensor

    Shape:
        Input: :math:`(C, H, W)` for numpy array
        Output: :math:`(N, C, H, W)`

    Returns:
        torch.Tensor (torch.float32): Transformed image tensor

    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor

def apply_transforms_v0(image, size=224):
    """Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of :math:`(N, C, H, W)` and
    is ready to be used as an input to neural networks.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.

    The plan is to to expand this to handle custom size/mean/std.

    Args:
        image (PIL.Image.Image or numpy array)
        size (int, optional, default=224): Desired size (width/height) of the
            output tensor

    Shape:
        Input: :math:`(C, H, W)` for numpy array
        Output: :math:`(N, C, H, W)`

    Returns:
        torch.Tensor (torch.float32): Transformed image tensor

    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor


def denormalize(tensor):
    """Reverses the normalisation on a tensor.

    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.

    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean

    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor

    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)

    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]

    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized[0], means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def standardize_and_clip(tensor, min_value=0.0, max_value=1.0):
    """Standardizes and clips input tensor.

    Standardize the input tensor (mean = 0.0, std = 1.0), ensures std is 0.1
    and clips it to values between min/max (default: 0.0/1.0).

    Args:
        tensor (torch.Tensor):
        min_value (float, optional, default=0.0)
        max_value (float, optional, default=1.0)

    Shape:
        Input: :math:`(C, H, W)`
        Output: Same as the input

    Return:
        torch.Tensor (torch.float32): Normalised tensor with values between
            [min_value, max_value]

    """

    tensor = tensor.detach().cpu()

    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        std += 1e-7

    standardized = tensor.sub(mean).div(std).mul(0.1)
    clipped = standardized.add(0.5).clamp(min_value, max_value)

    return clipped


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.

    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.

    Args:
        tensor (torch.Tensor, torch.float32): Image tensor

    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively

    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)

    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image

    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


def visualize(input_, gradients, save_path=None, cmap='viridis', alpha=0.7):

    """ Method to plot the explanation.

        # Arguments
            input_: Tensor. Original image.
            gradients: Tensor. Saliency map result.
            save_path: String. Defaults to None.
            cmap: Defaults to be 'viridis'.
            alpha: Defaults to be 0.7.

    """

    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))

    subplots = [
        ('Input image', [(input_, None, None)]),
        ('Saliency map across RGB channels', [(gradients, None, None)]),
        ('Overlay', [(input_, None, None), (gradients, cmap, alpha)])
    ]

    num_subplots = len(subplots)

    fig = plt.figure(figsize=(16, 3))

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()

        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)

        ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)


def basic_visualize(input_, gradients, save_path=None, weight=None, cmap='viridis', alpha=0.7):

    """ Method to plot the explanation.

        # Arguments
            input_: Tensor. Original image.
            gradients: Tensor. Saliency map result.
            save_path: String. Defaults to None.
            cmap: Defaults to be 'viridis'.
            alpha: Defaults to be 0.7.

    """
    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))

    subplots = [
        ('Saliency map across RGB channels', [(gradients, None, None)]),
        ('Overlay', [(input_, None, None), (gradients, cmap, alpha)])
    ]

    num_subplots = len(subplots)

    fig = plt.figure(figsize=(4, 4))

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()

        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)

    if save_path is not None:
        plt.savefig(save_path)

def basic_visualize_separate(input_, gradients, save_path=None, weight=None, cmap='viridis', alpha=0.7):
    """
    Save the input and overlayed gradient map images separately with no background or borders.
    
    # Arguments
        input_: Tensor. Original image.
        gradients: Tensor. Saliency map result.
        save_path: String. Base name for output images (without extension).
        cmap: Defaults to 'viridis'.
        alpha: Defaults to 0.7.
    """
    input_img = format_for_plotting(denormalize(input_))
    grad_img = format_for_plotting(standardize_and_clip(gradients))

    # Save saliency map with no axes, no padding, transparent background
    if save_path is not None:
        saliency_path = f"{save_path}_saliency.png"
        plt.figure()
        plt.axis('off')
        plt.imshow(grad_img)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(saliency_path, transparent=True, facecolor='none', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save overlay image
        overlay_path = f"{save_path}_overlay.png"
        plt.figure()
        plt.axis('off')
        plt.imshow(input_img)
        plt.imshow(grad_img, cmap=cmap, alpha=alpha)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(overlay_path, transparent=True, facecolor='none', bbox_inches='tight', pad_inches=0)
        plt.close()

import matplotlib.pyplot as plt
import numpy as np

def basic_visualize_sequence(input_, gradients_list, save_path=None, cmap='viridis', alpha=0.7):
    """
    Plot multiple gradient maps (CAMs) in two rows:
      - Top row: Saliency maps alone
      - Bottom row: Overlay of saliency maps on the input image
    
    Both rows have no gaps, no background, and no whitespace.
    
    # Arguments
        input_: Tensor. Original input image tensor.
        gradients_list: List of Tensors. Multiple gradient maps to visualize.
        save_path: If provided, saves the figure to this path.
        cmap: Colormap for overlay.
        alpha: Transparency for overlay on the bottom row.
    """
    # Preprocess input image and saliency maps
    input_img = format_for_plotting(denormalize(input_))
    processed_gradients = [format_for_plotting(standardize_and_clip(g)) for g in gradients_list]

    num_maps = len(processed_gradients)
    # Create a figure with 2 rows and num_maps columns
    # Height is doubled since we have two rows now
    fig, axes = plt.subplots(2, num_maps, figsize=(num_maps*4, 8), facecolor='none')

    # Ensure axes is 2D even if num_maps=1
    if num_maps == 1:
        axes = np.array([axes])  # ensures axes[0], axes[1] exist

    # Remove all spacing between subplots
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    # Top row: Just saliency maps
    for i, grad in enumerate(processed_gradients):
        ax_top = axes[0, i] if num_maps > 1 else axes[0]
        ax_top.set_axis_off()
        ax_top.xaxis.set_major_locator(plt.NullLocator())
        ax_top.yaxis.set_major_locator(plt.NullLocator())
        ax_top.imshow(grad, cmap=cmap)

    # Bottom row: Overlay of input and saliency
    for i, grad in enumerate(processed_gradients):
        ax_bottom = axes[1, i] if num_maps > 1 else axes[1]
        ax_bottom.set_axis_off()
        ax_bottom.xaxis.set_major_locator(plt.NullLocator())
        ax_bottom.yaxis.set_major_locator(plt.NullLocator())
        ax_bottom.imshow(input_img)
        ax_bottom.imshow(grad, cmap=cmap, alpha=alpha)

    # Save figure with no background, no extra margins
    if save_path is not None:
        plt.savefig(save_path, transparent=True, facecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close()


def combine_maps(maps, operation='max'):
    """
    Combine multiple maps with a specified operation.
    
    # Arguments
        maps: List of torch.Tensor or np.array of shape [H, W].
              The maps should have the same spatial dimensions.
        operation: str, one of {'max', 'add', 'mul'}.
    
    # Returns
        combined_map: A single combined map as a torch.Tensor with the same shape as the inputs.
    """
    # Ensure maps are converted to CPU and float tensors
    maps = [m.cpu().float() for m in maps]
    
    # Convert to numpy arrays for convenience if needed
    # or remain in torch. Here we'll use torch operations.
    # Assume all maps are [H, W]
    combined = maps[0].clone()
    if operation == 'max':
        for m in maps[1:]:
            combined = torch.max(combined, m)
    elif operation == 'add':
        for m in maps[1:]:
            combined += m
    elif operation == 'mul':
        # start combined as ones if we want pure product
        combined = torch.ones_like(maps[0])
        for m in maps:
            combined *= m
    else:
        raise ValueError("Operation must be one of 'max', 'add', or 'mul'.")
    
    return combined

def visualize_colored_maps(input_, maps, save_path=None, colors=None, alpha=0.5):
    """
    Visualize multiple maps by assigning each map a different color and overlaying them.
    
    # Arguments
        input_: Tensor. Original input image tensor (C,H,W).
        maps: List of torch.Tensor maps (H,W) to visualize.
        save_path: String. If provided, saves the figure to this path.
        colors: List of RGB tuples. If None, uses a default set.
        alpha: Float. Transparency for overlay.
    """
    if colors is None:
        colors = [
            (1, 0, 0),    # red
            (0, 1, 0),    # green
            (0, 0, 1),    # blue
            (1, 1, 0),    # yellow
            (1, 0, 1),    # magenta
            (0, 1, 1),    # cyan
            (1, 0.5, 0),  # orange
            (0.5, 0, 0.5) # purple
        ]

    # Convert input image and maps to NumPy arrays
    input_img = format_for_plotting(denormalize(input_))
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    input_img = input_img.astype(np.float32)

    processed_maps = []
    for m in maps:
        pm = format_for_plotting(standardize_and_clip(m))
        if isinstance(pm, torch.Tensor):
            pm = pm.detach().cpu().numpy()
        pm = pm.astype(np.float32)
        processed_maps.append(pm)

    # Start with the input image as the base
    composite = input_img.copy()  # H,W,3

    # Blend each colored map onto the image
    h, w, _ = composite.shape
    for i, pm in enumerate(processed_maps):
        color = colors[i % len(colors)]
        colored_map = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            colored_map[:, :, c] = pm * color[c]

        # Alpha blending
        composite = (1 - alpha) * composite + alpha * colored_map

    # Clip to [0,1]
    composite = np.clip(composite, 0, 1)

    # Save or display
    plt.figure()
    plt.axis('off')
    plt.imshow(composite)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if save_path is not None:
        plt.savefig(save_path, transparent=True, facecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close()

def basic_visualize_combination_progression(input_, maps, operation='max', save_path=None, cmap='viridis', alpha=0.7):
    """
    Visualize how combining maps iteratively changes the final output.
    
    For N maps:
      - Column 1: just map 1
      - Column 2: combined(map1, map2)
      - Column 3: combined(map1, map2, map3)
      ... and so forth until all maps are combined.
    
    In each column, we show two rows:
      Top row: The standalone combined saliency map at this stage
      Bottom row: The combined map overlayed on the input image
    
    # Arguments
        input_: Torch tensor for the input image (C,H,W).
        maps: List of Torch Tensors for saliency maps (H,W).
        operation: str in {'max', 'add', 'mul'} determining how to combine maps.
        save_path: If provided, the figure is saved.
        cmap: Colormap for the saliency maps.
        alpha: Transparency for the overlay.
    """
    # Ensure maps are CPU float tensors
    maps = [m.cpu().float() for m in maps]
    
    # Preprocess input image to numpy
    input_img = format_for_plotting(denormalize(input_))
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    input_img = input_img.astype(np.float32)

    # Function to combine maps incrementally
    def combine_op(current, new):
        if operation == 'max':
            return torch.max(current, new)
        elif operation == 'add':
            return current + new
        elif operation == 'mul':
            return current * new
        else:
            raise ValueError("Operation must be 'max', 'add', or 'mul'.")

    # Start combination with the first map
    combined_map = maps[0].clone()

    num_maps = len(maps)
    fig, axes = plt.subplots(2, num_maps, figsize=(num_maps*4, 8), facecolor='none')

    if num_maps == 1:
        axes = np.array([axes])  # Make sure indexing works uniformly

    # Remove spacing
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    for i in range(num_maps):
        if i > 0:
            # Combine next map
            combined_map = combine_op(combined_map, maps[i])

        # Process the combined map for display
        combined_np = format_for_plotting(standardize_and_clip(combined_map))
        if isinstance(combined_np, torch.Tensor):
            combined_np = combined_np.detach().cpu().numpy()
        combined_np = combined_np.astype(np.float32)

        # Top row: just the combined saliency map
        ax_top = axes[0, i] if num_maps > 1 else axes[0]
        ax_top.set_axis_off()
        ax_top.xaxis.set_major_locator(plt.NullLocator())
        ax_top.yaxis.set_major_locator(plt.NullLocator())
        ax_top.imshow(combined_np, cmap=cmap)

        # Bottom row: overlay on input
        ax_bottom = axes[1, i] if num_maps > 1 else axes[1]
        ax_bottom.set_axis_off()
        ax_bottom.xaxis.set_major_locator(plt.NullLocator())
        ax_bottom.yaxis.set_major_locator(plt.NullLocator())
        ax_bottom.imshow(input_img)
        ax_bottom.imshow(combined_np, cmap=cmap, alpha=alpha)

    if save_path is not None:
        plt.savefig(save_path, transparent=True, facecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close()

def basic_visualize_combination_progression_reverse(input_, maps, operation='max', save_path=None, cmap='viridis', alpha=0.7, reverse=True):
    """
    Visualize how combining maps iteratively changes the final output, starting from the last map.
    
    For N maps:
      - Column 1: just the last map (e.g., map N)
      - Column 2: combined(map N, map N-1)
      - Column 3: combined(map N, map N-1, map N-2)
      ... and so forth until all maps are combined.

    Each column shows two rows:
      Top row: The standalone combined saliency map at this stage.
      Bottom row: The combined map overlayed on the input image.
    
    # Arguments
        input_: Torch tensor for the input image (C,H,W).
        maps: List of Torch Tensors for saliency maps (H,W).
        operation: str in {'max', 'add', 'mul'} determining how to combine maps.
        save_path: If provided, the figure is saved.
        cmap: Colormap for the saliency maps.
        alpha: Transparency for the overlay.
    """
    # Ensure maps are CPU float tensors
    maps = [m.cpu().float() for m in maps]

    # Reverse the list so that maps[-1] (the last map) is now at index 0
    maps = maps[::-1]

    # Preprocess input image to numpy
    input_img = format_for_plotting(denormalize(input_))
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
    input_img = input_img.astype(np.float32)

    # Define combination operation
    def combine_op(current, new):
        if operation == 'max':
            return torch.max(current, new)
        elif operation == 'add':
            return current + new
        elif operation == 'mul':
            return current * new
        else:
            raise ValueError("Operation must be 'max', 'add', or 'mul'.")

    num_maps = len(maps)

    # Start with the last map alone (now maps[0] after reversing)
    combined_map = maps[0].clone()

    fig, axes = plt.subplots(2, num_maps, figsize=(num_maps*4, 8), facecolor='none')

    if num_maps == 1:
        axes = np.array([axes])  # Ensure indexing works uniformly

    # Remove spacing
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    # Iterate through maps (already reversed)
    for i in range(num_maps):
        if i > 0:
            # Combine next map backward
            combined_map = combine_op(combined_map, maps[i])

        # Process the combined map for display
        combined_np = format_for_plotting(standardize_and_clip(combined_map))
        if isinstance(combined_np, torch.Tensor):
            combined_np = combined_np.detach().cpu().numpy()
        combined_np = combined_np.astype(np.float32)

        if reverse:
            col = num_maps - 1 - i
        else:
            col = i

        # Top row: just the combined saliency map
        ax_top = axes[0, col] if num_maps > 1 else axes[0]
        ax_top.set_axis_off()
        ax_top.xaxis.set_major_locator(plt.NullLocator())
        ax_top.yaxis.set_major_locator(plt.NullLocator())
        ax_top.imshow(combined_np, cmap=cmap)

        # Bottom row: overlay on input
        ax_bottom = axes[1, col] if num_maps > 1 else axes[1]
        ax_bottom.set_axis_off()
        ax_bottom.xaxis.set_major_locator(plt.NullLocator())
        ax_bottom.yaxis.set_major_locator(plt.NullLocator())
        ax_bottom.imshow(input_img)
        ax_bottom.imshow(combined_np, cmap=cmap, alpha=alpha)

    if save_path is not None:
        plt.savefig(save_path, transparent=True, facecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close()


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if target_layer_name is None:
        target_layer_name = 'layer4'

    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    if target_layer_name is None:
        target_layer_name = 'features'

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if target_layer_name is None:
        target_layer_name = 'features'

    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if target_layer_name is None:
        target_layer_name = 'features_29'

    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++

        Args:
            - **arch - **: default torchvision densenet models
            - **target_layer_name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
                target_layer_name = 'features_12'
                target_layer_name = 'features_12_expand3x3'
                target_layer_name = 'features_12_expand3x3_activation'

        Return:
            target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if target_layer_name is None:
        target_layer_name = 'features'

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + '_' + hierarchy[3]]

    return target_layer


def find_googlenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++

        Args:
            - **arch - **: default torchvision googlenet models
            - **target_layer_name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
                target_layer_name = 'inception5b'

        Return:
            target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if target_layer_name is None:
        target_layer_name = 'features'

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + '_' + hierarchy[3]]

    return target_layer


def find_mobilenet_layer(arch, target_layer_name):
    """Find mobilenet layer to calculate GradCAM and GradCAM++

        Args:
            - **arch - **: default torchvision googlenet models
            - **target_layer_name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
                target_layer_name = 'features'

        Return:
            target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if target_layer_name is None:
        target_layer_name = 'features'

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + '_' + hierarchy[3]]

    return target_layer


def find_shufflenet_layer(arch, target_layer_name):
    """Find mobilenet layer to calculate GradCAM and GradCAM++

        Args:
            - **arch - **: default torchvision googlenet models
            - **target_layer_name (str) - **: the name of layer with its hierarchical information. please refer to usages below.
                target_layer_name = 'conv5'

        Return:
            target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if target_layer_name is None:
        target_layer_name = 'features'

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + '_' + hierarchy[3]]

    return target_layer


def find_layer(arch, target_layer_name):
    """Find target layer to calculate CAM.

        : Args:
            - **arch - **: Self-defined architecture.
            - **target_layer_name - ** (str): Name of target class.

        : Return:
            - **target_layer - **: Found layer. This layer will be hooked to get forward/backward pass information.
    """

    if target_layer_name.split('_') not in arch._modules.keys():
        raise Exception("Invalid target layer name.")
    target_layer = arch._modules[target_layer_name]
    return target_layer
