#!/usr/bin/env python3
"""
Train a segmentation model on prepared patches.

This script trains a U-Net on patches prepared by prepare.py.
It uses Binary Cross Entropy (BCE) and Dice loss with early stopping based on validation IoU.

Usage:
    python train.py
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNet, SolarFarmDataset

# Settings
TRAIN_PATCHES_DIR = "patches/train"
VAL_PATCHES_DIR = "patches/val"
MODEL_PATH = "models/solar_unet.pth"
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 10
CROP_SIZE = 128  # Model input size (patches may be larger for random crop augmentation)


def dice_bce_loss(pred, target, bce_weight=0.5, label_smoothing=0.1):
    """Combined BCE and Dice loss with label smoothing."""
    # Apply label smoothing: smooth targets from {0, 1} to {smooth, 1-smooth}
    target_smooth = target * (1 - label_smoothing) + (1 - target) * label_smoothing
    bce = F.binary_cross_entropy_with_logits(pred, target_smooth)
    pred_sig = torch.sigmoid(pred)
    # Use original (non-smoothed) targets for Dice to preserve IoU semantics
    intersection = (pred_sig * target).sum()
    dice = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)
    return bce_weight * bce + (1 - bce_weight) * dice


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_intersection = 0
    total_union = 0

    for embeddings, masks in tqdm(loader, desc="Training", leave=False):
        embeddings = embeddings.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = dice_bce_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Global IoU accumulation
        with torch.no_grad():
            pred_binary = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (pred_binary * masks).sum().item()
            union = pred_binary.sum().item() + masks.sum().item() - intersection
            total_intersection += intersection
            total_union += union

    avg_loss = total_loss / len(loader)
    global_iou = total_intersection / (total_union + 1e-6)
    return avg_loss, global_iou


def save_validation_predictions(model, val_dataset, device, output_dir: Path):
    """Save validation predictions as GeoTIFFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Saving predictions"):
            embedding, _ = val_dataset[idx]
            embedding = embedding.unsqueeze(0).to(device)

            output = model(embedding)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()

            # Get CRS and transform from corresponding mask.tif
            emb_path = val_dataset.patch_files[idx]
            mask_tif_path = emb_path.parent / emb_path.name.replace("_emb.npy", "_mask.tif")

            with rasterio.open(mask_tif_path) as src:
                crs = src.crs
                transform = src.transform

            # Save prediction
            patch_name = emb_path.stem.replace("_emb", "")
            with rasterio.open(
                output_dir / f"{patch_name}_pred.tif",
                "w",
                driver="GTiff",
                height=pred.shape[0],
                width=pred.shape[1],
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(pred, 1)


def validate(model, loader, device):
    """Validate and return metrics (global accumulation matching evaluate.py)."""
    model.eval()
    total_loss = 0
    total_intersection = 0
    total_pred_sum = 0
    total_target_sum = 0

    with torch.no_grad():
        for embeddings, masks in tqdm(loader, desc="Validating", leave=False):
            embeddings = embeddings.to(device)
            masks = masks.to(device)

            outputs = model(embeddings)
            loss = dice_bce_loss(outputs, masks)

            total_loss += loss.item()

            # Global metric accumulation
            pred_binary = (torch.sigmoid(outputs) > 0.5).float()
            total_intersection += (pred_binary * masks).sum().item()
            total_pred_sum += pred_binary.sum().item()
            total_target_sum += masks.sum().item()

    avg_loss = total_loss / len(loader)
    eps = 1e-6
    total_union = total_pred_sum + total_target_sum - total_intersection
    global_metrics = {
        "iou": total_intersection / (total_union + eps),
        "precision": total_intersection / (total_pred_sum + eps),
        "recall": total_intersection / (total_target_sum + eps),
    }
    return avg_loss, global_metrics


def main():
    """Train the model."""
    train_patches_path = Path(TRAIN_PATCHES_DIR)
    val_patches_path = Path(VAL_PATCHES_DIR)
    model_save_path = Path(MODEL_PATH)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets (val uses training stats for normalization)
    print("Loading datasets...")
    train_dataset = SolarFarmDataset(train_patches_path, augment=True, crop_size=CROP_SIZE)
    val_dataset = SolarFarmDataset(val_patches_path, stats_path=train_patches_path / "stats.json", crop_size=CROP_SIZE)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model
    model = UNet.nano(in_channels=128, out_channels=1)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop with early stopping
    best_iou = 0
    patience_counter = 0

    print(f"Training for up to {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}"
        )

        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": model.config,
                "iou": best_iou,
                "mean": train_dataset.mean.tolist(),
                "std": train_dataset.std.tolist(),
            }, model_save_path)
            print(f"  -> Saved best model (IoU: {best_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()

        if patience_counter >= PATIENCE:
            print(f"Early stopping (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {model_save_path}")

    # Load best model and save validation predictions as GeoTIFFs
    print("\nSaving validation predictions...")
    checkpoint = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_validation_predictions(model, val_dataset, device, Path("tmp/train_tifs/val_predictions"))
    print(f"Validation predictions saved to tmp/train_tifs/val_predictions/")


if __name__ == "__main__":
    main()
