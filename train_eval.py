import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from dataset import ImageNetLocDataset
from camformerv2 import CAMFormerModule
import wandb

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

@torch.no_grad()
def compute_miou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.1):
    """
    Compute the IoU for a batch of predicted masks vs ground truth masks.
    pred_mask and gt_mask are each [B, 224, 224] in shape.
    We threshold pred_mask > threshold to get a binary mask.
    Returns mean IoUs in the batch.
    """
    # Binarize predicted mask
    pred_bin = (pred_mask > threshold).float()

    # Intersection and union
    intersection = (pred_bin * gt_mask).sum(dim=(-1, -2))  # sum over H,W
    union = pred_bin.sum(dim=(-1, -2)) + gt_mask.sum(dim=(-1, -2)) - intersection

    iou = intersection / (union + 1e-8)  # avoid division by zero
    return torch.mean(iou)  # IoU per batch

def train_one_epoch(model, dataloader, optimizer, device, epoch, log_interval=100):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (imgs, bboxes, gt_masks, img_paths, image_ids) in enumerate(tqdm(dataloader), start=1):
        imgs = imgs.to(device)
        gt_masks = gt_masks.to(device)

        # Forward pass
        output_dict = model(imgs, class_idx=None, do_gradcam=True)

        masks = output_dict["cam"]
        loss = torch.nn.functional.mse_loss(masks, gt_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log every `log_interval` batches
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            wandb.log({
                "train_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx,
                "progress": batch_idx / num_batches  # Fraction of epoch completed
            })
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, dataloader, device, epoch, log_interval=100):
    model.eval()
    total_iou = 0.0
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (imgs, bboxes, gt_masks, img_paths, image_ids) in enumerate(tqdm(dataloader), start=1):
        imgs = imgs.to(device)
        gt_masks = gt_masks.to(device)

        output_dict = model(imgs, class_idx=None, do_gradcam=True, eval=True)
        #pred_boxes = output_dict["bbox"]  # [B,4]

        masks = output_dict["cam"] 
        iou = compute_miou(masks, gt_masks)
        total_iou += iou

        loss = torch.nn.functional.mse_loss(masks, gt_masks)
        total_loss += loss.item()

        # Log every `log_interval` batches
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            wandb.log({
                "val_loss": loss.item(),
                "val_iou": iou,
                "epoch": epoch,
                "val_batch": batch_idx,
                "val_progress": batch_idx / num_batches  # Fraction of epoch completed
            })

    return total_iou / num_batches, total_loss / num_batches


def main():
    wandb.init(project="CAMFormer", name="My-CAMFormer-Run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join("D:","imagenet","imagenet-object-localization-challenge","ILSVRC")

    # Example: train_loc.txt and val.txt are provided in ILSVRC/ImageSets/CLS-LOC
    train_dataset = ImageNetLocDataset(root, split="train", txt_file="train_loc.txt")
    val_dataset   = ImageNetLocDataset(root, split="val",   txt_file="val.txt")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=4)
    
    layer_names = ["features.4", "features.9", "features.16", "features.23", "features.30"] # all max pool layers

    model = CAMFormerModule(
        layer_names=layer_names,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=1000
    )
    model.to(device)

    learning_rate = 1e-4
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    # Start training
    epochs = 10

    wandb.init(project="CAMFormer", name="My-CAMFormer-Run", config={
    "learning_rate": learning_rate,
    "architecture": "VGG",
    "dataset": "ImageNet",
    "epochs": epochs,
    })

    log_interval = 100
    best_val_loss = float("inf")

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, log_interval)
        val_iou, val_loss = evaluate(model, val_loader, device, epoch, log_interval)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_camformer.pth")
            print(f"✅ Best model saved with Val Loss: {val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "camformer_final.pth")

if __name__ == "__main__":
    main()
