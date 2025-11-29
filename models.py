"""
Shared model components for solar farm segmentation.

This module provides:
- UNetSmall: A compact U-Net architecture for segmentation
- SolarFarmDataset: PyTorch dataset for loading patches
- dice_bce_loss: Combined BCE + Dice loss function
- compute_metrics: Segmentation metrics (IoU, Dice, Precision, Recall, F1)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset


# =============================================================================
# Model
# =============================================================================


class DoubleConv(nn.Module):
    """Two convolution layers with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetSmall(nn.Module):
    """Small U-Net for segmentation (3 encoder levels)."""

    def __init__(self, in_channels: int = 128, out_channels: int = 1, base_features: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_features * 4, base_features * 8)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)

        self.up3 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 2, base_features)

        # Output
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        bn = self.bottleneck(self.pool3(enc3))

        dec1 = self.dec1(torch.cat([self.up1(bn), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec1), enc2], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec2), enc1], dim=1))

        return torch.sigmoid(self.out_conv(dec3))


# =============================================================================
# Dataset
# =============================================================================


class SolarFarmDataset(Dataset):
    """Dataset for loading solar farm segmentation patches."""

    def __init__(self, patches_dir: Path, augment: bool = False):
        self.patches_dir = Path(patches_dir)
        self.augment = augment

        # Find all patch files
        self.patch_files = sorted(self.patches_dir.glob("*_emb.npy"))
        if not self.patch_files:
            raise ValueError(f"No patches found in {patches_dir}")

        # Compute normalization stats from a sample
        self._compute_stats()

    def _compute_stats(self):
        """Compute per-channel mean and std from a sample of patches."""
        n_samples = min(100, len(self.patch_files))
        indices = np.random.choice(len(self.patch_files), n_samples, replace=False)

        samples = [np.load(self.patch_files[i]) for i in indices]
        data = np.stack(samples, axis=0)  # (N, H, W, C)

        self.mean = data.mean(axis=(0, 1, 2))  # (C,)
        self.std = data.std(axis=(0, 1, 2)) + 1e-8  # (C,)

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        emb_path = self.patch_files[idx]
        mask_path = emb_path.parent / emb_path.name.replace("_emb.npy", "_mask.npy")

        embedding = np.load(emb_path)  # (H, W, 128)
        mask = np.load(mask_path)  # (H, W)

        # Normalize
        embedding = (embedding - self.mean) / self.std

        # Convert to tensors: (H, W, C) -> (C, H, W)
        embedding = torch.from_numpy(embedding).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                embedding = torch.flip(embedding, [2])
                mask = torch.flip(mask, [2])
            if np.random.random() > 0.5:
                embedding = torch.flip(embedding, [1])
                mask = torch.flip(mask, [1])

        return embedding, mask


# =============================================================================
# Loss Function
# =============================================================================


def dice_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE and Dice loss."""
    # BCE loss
    bce = F.binary_cross_entropy(pred, target)

    # Dice loss
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = 1 - (2 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    return 0.5 * bce + 0.5 * dice


# =============================================================================
# Metrics
# =============================================================================


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute segmentation metrics: IoU, Dice, Precision, Recall, F1."""
    pred_binary = (pred > threshold).float()

    pred_flat = pred_binary.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()

    # IoU
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Dice
    dice = (2 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average="binary", zero_division=0
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
