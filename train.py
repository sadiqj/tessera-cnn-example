#!/usr/bin/env python3
"""
Train a segmentation model on prepared patches.

This script trains a small U-Net on patches prepared by prepare.py.
It uses BCE + Dice loss and early stopping based on validation IoU.

Usage:
    python train.py --patches patches/train --model models/solar_unet.pth
    python train.py --patches patches/train --model models/solar_unet.pth --epochs 100
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models import UNetSmall, SolarFarmDataset, dice_bce_loss, compute_metrics

# Settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 10
VAL_SPLIT = 0.1


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metrics_list = []

    for embeddings, masks in tqdm(loader, desc="Training", leave=False):
        embeddings = embeddings.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = dice_bce_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            metrics = compute_metrics(outputs, masks)
            metrics_list.append(metrics)

    avg_loss = total_loss / len(loader)
    avg_iou = np.mean([m["iou"] for m in metrics_list])
    return avg_loss, avg_iou


def validate(model, loader, device):
    """Validate and return metrics."""
    model.eval()
    total_loss = 0
    metrics_list = []

    with torch.no_grad():
        for embeddings, masks in tqdm(loader, desc="Validating", leave=False):
            embeddings = embeddings.to(device)
            masks = masks.to(device)

            outputs = model(embeddings)
            loss = dice_bce_loss(outputs, masks)

            total_loss += loss.item()
            metrics = compute_metrics(outputs, masks)
            metrics_list.append(metrics)

    avg_loss = total_loss / len(loader)
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    return avg_loss, avg_metrics


def main(patches_dir: str, model_path: str, epochs: int):
    """Train the model."""
    patches_path = Path(patches_dir)
    model_save_path = Path(model_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    full_dataset = SolarFarmDataset(patches_path, augment=True)

    # Split into train/val
    n_val = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model
    model = UNetSmall(in_channels=128, out_channels=1)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_iou = 0
    patience_counter = 0

    print(f"Training for up to {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}"
        )

        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "iou": best_iou,
            }, model_save_path)
            print(f"  -> Saved best model (IoU: {best_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--patches", default="patches/train", help="Patches directory")
    parser.add_argument("--model", default="models/solar_unet.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    args = parser.parse_args()

    main(args.patches, args.model, args.epochs)
