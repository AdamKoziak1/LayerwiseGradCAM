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
        # Construct the original image path
        img_path = os.path.join(self.root, "Data", "CLS-LOC", self.split, image_id)

        if self.use_precomputed:
            precomputed_path = img_path + ".pth"
            if not os.path.exists(precomputed_path):
                raise FileNotFoundError(f"Precomputed file not found: {precomputed_path}")
            data = torch.load(precomputed_path)
            # Return the loaded precomputed data along with the image path.
            return data, img_path
        else:
            # Load the image normally.
            img = Image.open(img_path + ".JPEG").convert("RGB")
            img = apply_transforms(img)
            # Parse XML for bounding boxes.
            ann_path = os.path.join(self.root, "Annotations", "CLS-LOC", self.split, image_id + ".xml")
            obj = None
            if os.path.exists(ann_path):
                tree = ET.parse(ann_path)
                root_xml = tree.getroot()
                obj = root_xml.find("object")
            if obj is None:
                bbox = torch.tensor([0, 0, 1, 1], dtype=torch.float)
                mask = torch.zeros((224, 224), dtype=torch.float32)
            else:
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)
                mask = torch.zeros((224, 224), dtype=torch.float32)
                # Clamp coordinates and create mask.
                xmin, ymin, xmax, ymax = bbox.long()
                xmin = xmin.clamp(0, 224)
                xmax = xmax.clamp(0, 224)
                ymin = ymin.clamp(0, 224)
                ymax = ymax.clamp(0, 224)
                mask[ymin:ymax, xmin:xmax] = 1.0
            # Return image, bbox, mask, and the original image path.

            return img, bbox, mask, img_path, image_id
