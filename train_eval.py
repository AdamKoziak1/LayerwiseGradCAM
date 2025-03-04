import os
import platform
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from dataset import ImageNetLocDataset
from camformer import CAMFormerModule
import wandb
from torch.amp import autocast
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAMFormer")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of Transformer layers")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension size for Transformer")
    parser.add_argument('--dropout_rate', type=float, default=0.3, help="Dropout rate for Transformer layers")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    return parser.parse_args()

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

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    optimizer.zero_grad()

    for batch_idx, (imgs, bboxes, gt_masks, img_paths, image_ids) in enumerate(tqdm(dataloader), start=1):
        imgs = imgs.to(device)
        gt_masks = gt_masks.to(device)

        with autocast(device_type='cuda', dtype=torch.bfloat16): 
            output_dict = model(imgs, class_idx=None, do_gradcam=True)
            masks = output_dict["cam"]
            loss = torch.nn.functional.mse_loss(masks, gt_masks)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_iou = 0.0
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (imgs, bboxes, gt_masks, img_paths, image_ids) in enumerate(tqdm(dataloader), start=1):
        imgs = imgs.to(device)
        gt_masks = gt_masks.to(device)

        with autocast(device_type='cuda', dtype=torch.bfloat16): 
            output_dict = model(imgs, class_idx=None, do_gradcam=True, eval=True)
            masks = output_dict["cam"] 

            iou = compute_miou(masks, gt_masks)
            total_iou += iou

            loss = torch.nn.functional.mse_loss(masks, gt_masks)
            total_loss += loss.item()
    return total_iou / num_batches, total_loss / num_batches

def get_root_path():
    """
    Automatically detect whether the script is running on Windows or Linux (Ubuntu/WSL)
    and set the appropriate dataset path.
    """
    if platform.system() == "Windows":
        root = os.path.join("D:", "imagenet", "imagenet-object-localization-challenge", "ILSVRC")
    else:  # Linux / WSL
        root = os.path.join("/mnt", "d", "imagenet", "imagenet-object-localization-challenge", "ILSVRC")
    
    return root


def main():
    args = parse_args()
    wandb.init(project="CAMFormer", config=vars(args))
    torch.cuda.init()
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    root = get_root_path()
    print(f"Using dataset root: {root}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_name = f"run_{wandb.run.id}"  # Use auto-generated name or run ID
    checkpoint_prefix = f"{checkpoint_dir}/{run_name}"  # e.g., "checkpoints/run-abc123_20240211-154530"


    # Example: train_loc.txt and val.txt are provided in ILSVRC/ImageSets/CLS-LOC
    train_dataset = ImageNetLocDataset(root, split="train", txt_file="train_loc.txt")
    val_dataset   = ImageNetLocDataset(root, split="val",   txt_file="val.txt")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    layer_names = ["features.4", "features.9", "features.16", "features.23", "features.30"] # all max pool layers

    model = CAMFormerModule(
        layer_names=layer_names,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_classes=1000,
    )
    model.to(device)

    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)

    epochs = args.epochs

    best_val_loss = float("inf")
    early_stopping_patience = 3
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_iou, val_loss = evaluate(model, val_loader, device, epoch)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = f"{checkpoint_prefix}_best.pth"
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"✅ Best model saved with Val Loss: {val_loss:.4f}")
            epochs_no_improve = 0  # reset early stopping counter
        else:
            epochs_no_improve += 1
            print(f"No improvement in val loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    final_checkpoint_path = f"{checkpoint_prefix}_final.pth"
    torch.save(model.state_dict(), final_checkpoint_path)

if __name__ == "__main__":
    main()
