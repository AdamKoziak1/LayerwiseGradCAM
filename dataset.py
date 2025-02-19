import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import *

class ImageNetLocDataset(Dataset):
    """
    Minimal ImageNet Localisation dataset loader.
    Returns either raw images (with bbox and mask) or precomputed features,
    along with the image path. To toggle precomputed loading, set use_precomputed=True
    and provide the precomputed_dir.
    """
    def __init__(self, root, split="train", txt_file="train_loc.txt", use_precomputed=False):
        """
        :param root: Path to the ILSVRC folder.
        :param split: "train" or "val".
        :param txt_file: e.g. "train_loc.txt" or "val.txt".
        :param use_precomputed: If True, load precomputed features instead of images.
        :param precomputed_dir: Directory where precomputed files are stored.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.use_precomputed = use_precomputed
        # e.g. ILSVRC/ImageSets/CLS-LOC/train_loc.txt
        txt_path = os.path.join(root, "ImageSets", "CLS-LOC", txt_file)
        with open(txt_path, "r") as f:
            # Each line has an image identifier, e.g. 'n01440764_18' or possibly with subdirectories
            self.image_ids = [line.split()[0].strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # Construct the original image path.
        img_path = os.path.join(self.root, "Data", "CLS-LOC", self.split, image_id)

        # TODO update for new returns
        if self.use_precomputed:
            precomputed_path = img_path + ".pth"
            if not os.path.exists(precomputed_path):
                raise FileNotFoundError(f"Precomputed file not found: {precomputed_path}")
            data = torch.load(precomputed_path)
            return data, img_path
        else:
            # Load the original image.
            orig_img = Image.open(img_path + ".JPEG").convert("RGB")
            orig_w, orig_h = orig_img.size

            # Define the crop size (the final output size from our transforms)
            crop_size = 224

            # --- Compute what transforms.Resize(crop_size) does ---
            # When passing an integer to transforms.Resize, the smaller edge is matched to that value.
            if orig_w < orig_h:
                resized_w = crop_size
                resized_h = int(orig_h * (crop_size / orig_w))
            else:
                resized_h = crop_size
                resized_w = int(orig_w * (crop_size / orig_h))
            # --- Compute centre crop offset ---
            # The centre crop will extract a (crop_size x crop_size) region from the centre.
            offset_x = (resized_w - crop_size) / 2
            offset_y = (resized_h - crop_size) / 2

            # --- Parse XML for bounding box ---
            ann_path = os.path.join(self.root, "Annotations", "CLS-LOC", self.split, image_id + ".xml")
            obj = None
            if os.path.exists(ann_path):
                tree = ET.parse(ann_path)
                root_xml = tree.getroot()
                obj = root_xml.find("object")
            if obj is None:
                # If no annotation, default to full image.
                bbox = torch.tensor([0, 0, crop_size, crop_size], dtype=torch.float)
                mask = torch.zeros((crop_size, crop_size), dtype=torch.float32)
            else:
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # --- Update the bounding box for the transformed image ---
                # First, scale from original image to resized image coordinates.
                scale_x = resized_w / orig_w
                scale_y = resized_h / orig_h
                resized_bbox = [xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y]
                # Then, adjust for centre crop offset.
                cropped_bbox = [
                    resized_bbox[0] - offset_x,
                    resized_bbox[1] - offset_y,
                    resized_bbox[2] - offset_x,
                    resized_bbox[3] - offset_y,
                ]
                # Clamp the bounding box to [0, crop_size].
                cropped_bbox[0] = max(cropped_bbox[0], 0)
                cropped_bbox[1] = max(cropped_bbox[1], 0)
                cropped_bbox[2] = min(cropped_bbox[2], crop_size)
                cropped_bbox[3] = min(cropped_bbox[3], crop_size)
                bbox = torch.tensor(cropped_bbox, dtype=torch.float)

                # Create the binary mask from the updated bbox.
                mask = torch.zeros((crop_size, crop_size), dtype=torch.float32)
                x1, y1, x2, y2 = bbox.long()
                mask[y1:y2, x1:x2] = 1.0

            # Now apply the transforms to get the final image.
            # Note: apply_transforms internally does Resize(crop_size) followed by CenterCrop(crop_size).
            img = apply_transforms(orig_img, size=crop_size)
            
            # Return the transformed image, scaled & cropped bbox, mask, image path and image id.
            return img, bbox, mask, img_path, image_id
