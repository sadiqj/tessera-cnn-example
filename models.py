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


class UNet(nn.Module):
    """Configurable U-Net for segmentation.

    Args:
        in_channels: Number of input channels (default: 128 for GeoTessera embeddings)
        out_channels: Number of output classes (default: 1 for binary segmentation)
        depth: Number of encoder/decoder stages, range [2, 5] (default: 3)
        base_features: Number of features in first encoder layer (default: 32)
        dropout: Dropout rate applied after bottleneck (default: 0.3)

    Channel progression follows: base_features * 2^level pattern.
    For depth=3, base_features=32: 32 -> 64 -> 128 -> 256 (bottleneck)
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 1,
        depth: int = 3,
        base_features: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        if not 2 <= depth <= 5:
            raise ValueError(f"depth must be in [2, 5], got {depth}")

        # Store config for serialization
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._depth = depth
        self._base_features = base_features
        self._dropout = dropout

        # Channel progression: base_features * 2^level
        # For depth=3: [32, 64, 128, 256] (last is bottleneck)
        channels = [base_features * (2**i) for i in range(depth + 1)]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels[:-1]:  # All except bottleneck
            self.encoders.append(DoubleConv(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = ch

        # Bottleneck
        self.bottleneck = DoubleConv(channels[-2], channels[-1])
        self.dropout_layer = nn.Dropout2d(dropout)

        # Decoder (reverse order)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = channels[i + 1]
            out_ch = channels[i]
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                )
            )
            # After concat with skip: out_ch * 2
            self.decoders.append(DoubleConv(out_ch * 2, out_ch))

        # Output
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder pass - store skip connections
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout_layer(x)

        # Decoder pass - use skip connections in reverse order
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.out_conv(x)

    @property
    def config(self) -> dict:
        """Return model configuration for checkpoint saving."""
        return {
            "in_channels": self._in_channels,
            "out_channels": self._out_channels,
            "depth": self._depth,
            "base_features": self._base_features,
            "dropout": self._dropout,
        }

    @classmethod
    def from_config(cls, config: dict) -> "UNet":
        """Create model from configuration dict."""
        return cls(**config)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> tuple["UNet", dict]:
        """Load model from checkpoint file.

        Returns:
            Tuple of (model, checkpoint_dict) where checkpoint_dict contains
            additional data like mean, std, epoch, iou.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls.from_config(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint

    # Preset classmethods for common configurations
    @classmethod
    def nano(cls, **kwargs) -> "UNet":
        """Ultra-lightweight model (~42K params). Depth=2, base=8."""
        return cls(depth=2, base_features=8, **kwargs)

    @classmethod
    def tiny(cls, **kwargs) -> "UNet":
        """Tiny model (~148K params). Depth=2, base=16."""
        return cls(depth=2, base_features=16, **kwargs)

    @classmethod
    def small(cls, **kwargs) -> "UNet":
        """Small model (~2.2M params). Depth=3, base=32. Matches legacy UNetSmall."""
        return cls(depth=3, base_features=32, **kwargs)

    @classmethod
    def base(cls, **kwargs) -> "UNet":
        """Base model (~4.9M params). Depth=3, base=48."""
        return cls(depth=3, base_features=48, **kwargs)

    @classmethod
    def medium(cls, **kwargs) -> "UNet":
        """Medium model (~8.7M params). Depth=4, base=32."""
        return cls(depth=4, base_features=32, **kwargs)

    @classmethod
    def large(cls, **kwargs) -> "UNet":
        """Large model (~19.5M params). Depth=4, base=48."""
        return cls(depth=4, base_features=48, **kwargs)


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
