import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import *
from tqdm import tqdm

#################################################
# 1) DATASET
#################################################

class ImageNetLocDataset(Dataset):
    """
    Minimal ImageNet Localization dataset loader matching:
      ILSVRC/
       ┣ Annotations/CLS-LOC/{train,val}/some_id.xml
       ┣ Data/CLS-LOC/{train,val}/some_id.JPEG
       ┗ ImageSets/CLS-LOC/{train_loc.txt, val.txt}
    Each line in train_loc.txt or val.txt is an image identifier, e.g. 'n01440764_18'.
    """

    def __init__(self, root, split="train", txt_file="train_loc.txt"):
        """
        :param root: Path to the ILSVRC folder.
        :param split: "train" or "val".
        :param txt_file: e.g. "train_loc.txt" or "val.txt".
        """
        super().__init__()
        self.root = root
        self.split = split

        # e.g. ILSVRC/ImageSets/CLS-LOC/train_loc.txt
        txt_path = os.path.join(root, "ImageSets", "CLS-LOC", txt_file)


        with open(txt_path, "r") as f:
            # Each line has an image identifier, e.g. 'n01440764_18'
            self.image_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # Build paths for image and annotation
        # e.g. ILSVRC/Data/CLS-LOC/train/n01440764_18.JPEG
        #      ILSVRC/Annotations/CLS-LOC/train/n01440764_18.xml
        img_path = os.path.join(
            self.root, "Data", "CLS-LOC", self.split, image_id + ".JPEG"
        )
        ann_path = os.path.join(
            self.root, "Annotations", "CLS-LOC", self.split, image_id + ".xml"
        )

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = apply_transforms(img)

        # Parse XML for bounding boxes
        tree = ET.parse(ann_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            # If no bounding box found, yield a dummy box
            bbox = torch.tensor([0, 0, 1, 1], dtype=torch.float)
        else:
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)

        return img, bbox, image_id


class FAKEDataset(Dataset):
    """
    Minimal ImageNet Localization dataset loader matching:
      ILSVRC/
       ┣ Annotations/CLS-LOC/{train,val}/some_id.xml
       ┣ Data/CLS-LOC/{train,val}/some_id.JPEG
       ┗ ImageSets/CLS-LOC/{train_loc.txt, val.txt}
    Each line in train_loc.txt or val.txt is an image identifier, e.g. 'n01440764_18'.
    """

    def __init__(self):
        self.image_ids = [1,2,3,4,5,6,6,7,88,9,9,9,9,9,9,]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.tensor([0, 0, 1, 1], dtype=torch.float), idx




#################################################
# 2) MODEL: We assume you have a CAMFormer module
# with forward(...) returning output_dict with
# "bbox" of shape [B,4].
#################################################

from camformerv2 import CAMFormerModule

#################################################
# 3) TRAINING / VALIDATION HELPERS
#################################################


def box_l1_loss(pred_boxes, gt_boxes):
    """
    A simple L1 bounding box loss, each is shape [B,4].
    """
    return torch.nn.functional.l1_loss(pred_boxes, gt_boxes)

@torch.no_grad()
def compute_iou(pred_box, gt_box):
    """
    Compute IoU between two boxes: [xmin, ymin, xmax, ymax].
    """
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])
    if xB < xA or yB < yA:
        return 0.0
    interArea = (xB - xA) * (yB - yA)
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = boxAArea + boxBArea - interArea
    iou = interArea / union if union > 0 else 0.0
    return iou

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, bboxes, _ in dataloader:
        imgs = imgs.to(device)
        bboxes = bboxes.to(device)
        # Forward pass
        output_dict = model(imgs, class_idx=None, do_gradcam=True)
        pred_boxes = output_dict["bbox"]  # [B,4]

        print(f"Predicted bbox: {pred_boxes[0]}")
        print(f"Ground truth bbox: {bboxes[0]}")
        print(f"Predicted bbox requires grad: {pred_boxes.requires_grad}")
        print(f"Ground truth bbox requires grad: {bboxes.requires_grad}")

        # L1 loss on bounding boxes
        #loss = box_l1_loss(pred_boxes, bboxes)

        loss = torch.nn.functional.mse_loss(pred_boxes, bboxes)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    count = 0
    for imgs, bboxes, _ in dataloader:
        imgs = imgs.to(device)
        bboxes = bboxes.to(device)

        output_dict = model(imgs, class_idx=None, do_gradcam=True)
        pred_boxes = output_dict["bbox"]  # [B,4]

        # compute IoU
        for i in range(imgs.size(0)):
            iou = compute_iou(pred_boxes[i].tolist(), bboxes[i].tolist())
            total_iou += iou
        count += imgs.size(0)
    return total_iou / max(count, 1)

#################################################
# 4) MAIN SCRIPT
#################################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "/path/to/ILSVRC"  # Folder that has Data/ and Annotations/ subdirs

    # Example: train_loc.txt and val.txt are provided in ILSVRC/ImageSets/CLS-LOC
    #train_dataset = ImageNetLocDataset(root, split="train", txt_file="train_loc.txt")
    #val_dataset   = ImageNetLocDataset(root, split="val",   txt_file="val.txt")
    train_dataset = FAKEDataset()
    val_dataset   = FAKEDataset()

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4)
    
    layer_names = ["features.4", "features.9", "features.16", "features.23", "features.30"] # all max pool layers

    model = CAMFormerModule(
        layer_names=layer_names,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=1000
    )
    model.to(device)


    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Start training
    epochs = 5
    for epoch in tqdm(range(1, epochs+1)):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")

    # Save final model
    torch.save(model.state_dict(), "camformer_localization.pth")

if __name__ == "__main__":
    main()
