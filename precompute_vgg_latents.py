import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
from dataset import ImageNetLocDataset

# -----------------------------------------------------------------------------
# Helper to Find VGG Layer by Name
# -----------------------------------------------------------------------------
def find_vgg_layer(model, layer_name):
    return dict(model.named_modules())[layer_name]

# -----------------------------------------------------------------------------
# VGG Feature Extractor with Hooks
#
# This module wraps a pretrained VGG‐16 model, registers forward and backward hooks
# on the designated layers, and (when requested) performs a backward pass to capture
# gradients in a Grad‐CAM style.
# -----------------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_names):
        super().__init__()
        self.layer_names = layer_names
        self.activations = []
        self.gradients = []
        self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
        # Register hooks on the target layers
        for layer_name in self.layer_names:
            layer = find_vgg_layer(self.model, layer_name)
            layer.register_forward_hook(self.forward_hook)
            layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations.append(output.clone())

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].clone())

    def clear_hooks(self):
        self.activations = []
        self.gradients = []

    def forward(self, x, retain_graph=False):
        self.clear_hooks()
        logits = self.model(x)
        # Use the top predicted class for each sample
        predicted_class = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, predicted_class.unsqueeze(1), 1)
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=retain_graph)
        self.model.zero_grad()
        # Reverse the gradients to match the order of the hooks if needed
        self.gradients.reverse()
        return logits

# -----------------------------------------------------------------------------
# Precompute and Save Features
#
# This function iterates over the dataset, runs VGG (with a backward pass),
# detaches the activations/gradients, and saves each sample's data to its own
# .pth file in the specified output directory.
# -----------------------------------------------------------------------------
def precompute_and_save_features(root, split="val", txt_file="val.txt",
                                 batch_size=16, num_workers=4):
    dataset = ImageNetLocDataset(root, split=split, txt_file=txt_file, use_precomputed=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify the target layers (example: VGG-16 max-pool layers)
    layer_names = ["features.4", "features.9", "features.16", "features.23", "features.30"]
    extractor = VGGFeatureExtractor(layer_names)
    extractor.to(device)
    
    for imgs, bboxes, gt_masks, img_paths, image_ids in tqdm(dataloader, desc="Precomputing features"):
        imgs = imgs.to(device)
        # Forward pass (and backward pass to obtain gradients)
        _ = extractor(imgs, retain_graph=False)
        batch_size_current = imgs.size(0)
        for i in range(batch_size_current):
            sample_dict = {
                "act": [act[i].detach().cpu() for act in extractor.activations],
                "grad": [grad[i].detach().cpu() for grad in extractor.gradients],
                "bbox": bboxes[i].detach().cpu(),
                "gt_mask": gt_masks[i].detach().cpu()
            }
            out_path = img_paths[i] + ".pth"
            torch.save(sample_dict, out_path)
        extractor.clear_hooks()

if __name__ == "__main__":
    root = os.path.join("D:", "imagenet", "imagenet-object-localization-challenge", "ILSVRC")

    precompute_and_save_features(root, split="train", txt_file="train_loc.txt", batch_size=8, num_workers=4)

    precompute_and_save_features(root, split="val", txt_file="val.txt", batch_size=8, num_workers=4)

    precompute_and_save_features(root, split="test", txt_file="test.txt", batch_size=8, num_workers=4)
