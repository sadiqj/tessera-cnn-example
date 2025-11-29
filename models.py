"""
Shared model components for solar farm segmentation.

This module provides:
- UNetSmall: A compact U-Net architecture for segmentation
- SolarFarmDataset: PyTorch dataset for loading patches
- compute_metrics: Segmentation metrics (IoU, Dice, Precision, Recall)
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

    def __init__(self, in_channels: int = 128, out_channels: int = 1, base_features: int = 32, dropout: float = 0.3):
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
        self.dropout = nn.Dropout2d(dropout)

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
        bn = self.dropout(bn)

        dec1 = self.dec1(torch.cat([self.up1(bn), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec1), enc2], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec2), enc1], dim=1))

        return self.out_conv(dec3)


# =============================================================================
# Dataset
# =============================================================================


class SolarFarmDataset(Dataset):
    """Dataset for loading solar farm segmentation patches."""

    def __init__(
        self,
        patches_dir: Path,
        stats_path: Path | None = None,
        augment: bool = False,
        crop_size: int | None = None,
        mean: list | np.ndarray | None = None,
        std: list | np.ndarray | None = None,
    ):
        """
        Args:
            patches_dir: Directory containing patch .npy files
            stats_path: Path to stats.json. If None, uses patches_dir/stats.json.
                        For test/val sets, pass the training set's stats.json path.
                        Ignored if mean/std are provided explicitly.
            augment: If True, apply random augmentations (flips, rotations, random crops) during loading.
            crop_size: If set, randomly crop patches to this size during augmentation.
                      For validation/test, center crop is used instead.
            mean: Explicit mean values for normalization (overrides stats_path).
            std: Explicit std values for normalization (overrides stats_path).
        """
        self.augment = augment
        self.crop_size = crop_size
        self.patches_dir = Path(patches_dir)

        # Find all patch files
        self.patch_files = sorted(self.patches_dir.glob("*_emb.npy"))
        if not self.patch_files:
            raise ValueError(f"No patches found in {patches_dir}")

        # Load normalization stats: prioritize explicit values, fall back to stats.json
        if mean is not None and std is not None:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
        else:
            if stats_path is None:
                stats_path = self.patches_dir / "stats.json"
            else:
                stats_path = Path(stats_path)

            if not stats_path.exists():
                raise ValueError(f"stats.json not found at {stats_path}. Run prepare.py first.")

            with open(stats_path) as f:
                stats = json.load(f)

            self.mean = np.array(stats["mean"], dtype=np.float32)
            self.std = np.array(stats["std"], dtype=np.float32)

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        emb_path = self.patch_files[idx]
        mask_path = emb_path.parent / emb_path.name.replace("_emb.npy", "_mask.npy")

        embedding = np.load(emb_path)  # (H, W, 128)
        mask = np.load(mask_path)  # (H, W)

        # Apply augmentations (before normalization, on numpy arrays)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                embedding = np.flip(embedding, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                embedding = np.flip(embedding, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            # Random 90-degree rotation (0, 1, 2, or 3 times)
            k = np.random.randint(0, 4)
            if k > 0:
                embedding = np.rot90(embedding, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # Random crop (training) or center crop (validation/test)
        if self.crop_size is not None and embedding.shape[0] > self.crop_size:
            h, w = embedding.shape[:2]
            if self.augment:
                # Random crop for training
                y = np.random.randint(0, h - self.crop_size + 1)
                x = np.random.randint(0, w - self.crop_size + 1)
            else:
                # Center crop for validation/test
                y = (h - self.crop_size) // 2
                x = (w - self.crop_size) // 2
            embedding = embedding[y:y + self.crop_size, x:x + self.crop_size, :]
            mask = mask[y:y + self.crop_size, x:x + self.crop_size]

        # Normalize
        embedding = (embedding - self.mean) / self.std

        # Convert to tensors: (H, W, C) -> (C, H, W)
        embedding = torch.from_numpy(embedding).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        return embedding, mask


# =============================================================================
# Metrics
# =============================================================================


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute segmentation metrics: IoU, Dice, Precision, Recall.

    Args:
        pred: Predicted probabilities (already sigmoid-applied)
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
    """
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    pred_sum, target_sum = pred_binary.sum(), target.sum()
    union = pred_sum + target_sum - intersection

    eps = 1e-6
    return {
        "iou": float((intersection + eps) / (union + eps)),
        "dice": float((2 * intersection + eps) / (pred_sum + target_sum + eps)),
        "precision": float((intersection + eps) / (pred_sum + eps)),
        "recall": float((intersection + eps) / (target_sum + eps)),
    }
