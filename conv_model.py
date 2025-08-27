#!/usr/bin/env python3
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models import VGG16_Weights

def find_vgg_layer(model, layer_name):
    """
    Return the nn.Module corresponding to layer_name, e.g. "features.29" in vgg16.
    """
    return dict(model.named_modules())[layer_name]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CAMMaskModule(nn.Module):
    """
    This module uses a VGG16 backbone to extract intermediate activations.
    For each selected layer:
      1. The activation (shape [B, C, H, W]) is reshaped and processed with a 1D convolution 
         to produce a learned weighted combination of the channels.
      2. The resulting feature map is upsampled to a common spatial resolution.
    Then, the upsampled maps from all layers are concatenated and processed with a second 1D convolution 
    (applied across the layer dimension) followed by a sigmoid to generate a mask.
    This mask is applied to the original input image, which is then re‐passed through VGG16.
    A classification loss (cross‑entropy) on the masked image is computed using the true labels.
    """
    def __init__(self, layer_names, mask_size=(224, 224), num_classes=1000):
        super().__init__()

        # Load the pretrained VGG16 backbone and set to eval mode.
        backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
        self.model_arch = backbone.eval()

        self.layer_names = layer_names
        self.num_classes = num_classes
        self.mask_size = mask_size

        # Determine device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model_arch.to(self.device)

        # Containers for storing per-layer activations and gradients (for Grad-CAM, if needed)
        self.activations = []
        self.gradients = []

        # Step 1: Register temporary hooks to record each layer’s output shape.
        self.layer_shapes = {}
        hook_handles = []

        def shape_forward_hook(layer_id):
            def hook(module, inp, out):
                # out shape is [B, C, H, W]
                _, c, h, w = out.shape
                self.layer_shapes[layer_id] = (c, h, w)
            return hook

        for layer_n in self.layer_names:
            layer = find_vgg_layer(self.model_arch, layer_n)
            h = layer.register_forward_hook(shape_forward_hook(layer_n))
            hook_handles.append(h)

        # Run a dummy forward pass to record shapes.
        dummy_x = torch.randn(1, 3, 224, 224, device=self.device)
        _ = self.model_arch(dummy_x)
        for h in hook_handles:
            h.remove()

        # Step 2: Build per-layer 1D convolutions that combine the channels.
        # Each convolution maps from the layer’s C channels to 1.
        self.channel_convs = nn.ModuleList()
        for layer_n in self.layer_names:
            c, _, _ = self.layer_shapes[layer_n]
            conv = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1, dtype=torch.float)
            self.channel_convs.append(conv)

        # Step 3: Register hooks on the selected layers to capture activations and gradients.
        self.target_layers = []
        def forward_hook_fn(module, inp, out):
            # Save a detached copy of the activations.
            self.activations.append(out.clone().detach().requires_grad_(True))

        def backward_hook_fn(module, grad_in, grad_out):
            self.gradients.append(grad_out[0].clone().detach().requires_grad_(True))

        for layer_n in self.layer_names:
            lyr = find_vgg_layer(self.model_arch, layer_n)
            self.target_layers.append(lyr)
            lyr.register_forward_hook(forward_hook_fn)
            lyr.register_backward_hook(backward_hook_fn)

        # Step 4: Final 1D convolution to combine the upsampled feature maps across layers.
        # This convolution expects the per-layer maps to be stacked along the channel dimension.
        self.final_conv = nn.Conv2d(in_channels=len(self.layer_names),
                                    out_channels=1,
                                    kernel_size=1,
                                    dtype=torch.float)

        # Loss function for classification.
        self.criterion = nn.CrossEntropyLoss()

        self.channel_convs.to(self.device)
        self.final_conv.to(self.device)

    def forward(self, x, true_labels, do_gradcam=True, retain_graph=False):
        """
        :param x: Input image batch, shape [B, 3, H, W].
        :param true_labels: Tensor with true labels, shape [B].
        :param do_gradcam: If True, performs a backward pass to capture gradients.
        :param retain_graph: Whether to retain the graph after backward.
        :return: A dictionary containing the loss, mask, logits, accuracy, activations and gradients.
        """
        x = x.to(self.device)
        true_labels = true_labels.to(self.device)
        self._clear_hooks_storage()

        # First forward pass through VGG16 to capture activations.
        logits = self.model_arch(x)

        # Optionally, perform a backward pass to capture gradients (Grad-CAM style).
        if do_gradcam:
            # Use the provided true labels for the backward signal.
            oh = torch.zeros_like(logits)
            for b in range(x.size(0)):
                oh[b, true_labels[b]] = 1
            self.model_arch.zero_grad()
            logits.backward(gradient=oh, retain_graph=retain_graph)
            self.model_arch.zero_grad()
            # Reverse gradients list to align with activations.
            self.gradients.reverse()

        # For each selected layer, process the activation with a conv to combine channels.
        layer_masks = []
        for act, conv in zip(self.activations, self.channel_convs):
            # Apply the conv to produce a weighted combination of channels.
            combined = conv(act)  
            # Upsample the feature map to the common mask resolution.
            combined_up = F.interpolate(combined, size=self.mask_size, mode='bicubic', align_corners=False)
            layer_masks.append(combined_up)

        # Stack the upsampled feature maps along a new channel dimension.
        # Now shape is [B, N, H, W] where N is the number of selected layers.
        stacked_masks = torch.cat(layer_masks, dim=1)

        combined_layers = self.final_conv(stacked_masks)  
        # Normalise the mask using a sigmoid.
        mask = torch.sigmoid(combined_layers)

        # Use the mask to modulate the original input image.
        # The mask is broadcast over the three channels.
        x_masked = x * mask

        activations_out = copy.deepcopy(self.activations)
        gradients_out = copy.deepcopy(self.gradients)

        # Run VGG16 again on the masked image.

        logits_masked = self.model_arch(x_masked)
        loss = self.criterion(logits_masked, true_labels)

        # Compute accuracy for monitoring.
        preds = logits_masked.argmax(dim=1)
        correct = (preds == true_labels).sum().item()
        accuracy = correct / x.size(0)

        n = len(layer_names)
        print(n)
        output_dict = {
            "loss": loss,
            "mask": mask,
            "logits": logits_masked,
            "accuracy": accuracy,
            "activations": activations_out,
            "gradients": gradients_out
        }
        print(len(self.activations), len(self.gradients))
        print(len(output_dict["activations"]), len(output_dict["gradients"]))
        return output_dict

    def _clear_hooks_storage(self):
        """Helper: clear out stored activations and gradients between forward passes."""
        self.activations = []
        self.gradients = []

    def count_params(self):
        backbone_params = count_parameters(self.model_arch)
        channel_conv_params = count_parameters(self.channel_convs)
        final_conv_params = count_parameters(self.final_conv)
        total = backbone_params + channel_conv_params + final_conv_params
        print(f"Backbone params: {backbone_params}")
        print(f"Channel convs params: {channel_conv_params}")
        print(f"Final conv params: {final_conv_params}")
        print(f"Total params: {total}")
        return total

if __name__ == "__main__":
    # Define the selected layers for which activations will be captured.
    layer_names = ["features.4", "features.9", "features.16", "features.23", "features.30"]

    # Initialise the module.
    model = CAMMaskModule(layer_names=layer_names, mask_size=(224, 224), num_classes=1000)
    model.count_params()

    # Create a dummy batch of images.
    images = torch.randn(5, 3, 224, 224).to(model.device)
    # Create dummy true labels.
    true_labels = torch.randint(0, 1000, (5,)).to(model.device)

    # Forward pass.
    output_dict = model(images, true_labels, do_gradcam=True)
    mask = output_dict["mask"]
    logits = output_dict["logits"]
    loss = output_dict["loss"]
    accuracy = output_dict["accuracy"]

    print("Mask shape:", mask.shape)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Accuracy:", accuracy)
