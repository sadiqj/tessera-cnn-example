#!/usr/bin/env python3
"""
Evaluate the trained model on test/validation patches.

This script loads a trained model and evaluates it on a dataset split,
reporting IoU, Dice, Precision, and Recall scores.

It can also generate geo-referenced GeoTIFFs of predictions showing:
- Green: True Positives (correct detections)
- Red: False Negatives (missed solar farms)
- Blue: False Positives (false alarms)
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNetSmall, SolarFarmDataset

# Settings
MODEL_PATH = "models/solar_unet.pth"
CROP_SIZE = 128  # Model input size (patches may be larger)


def evaluate(patches_dir: str, output_dir: str = None, export_viz: bool = False):
    """Evaluate model on a dataset split and optionally export visualizations."""
    patches_path = Path(patches_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and extract normalization stats from checkpoint
    model_file = Path(MODEL_PATH)
    print(f"Loading model from {model_file}...")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    model = UNetSmall(in_channels=128, out_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model trained to epoch {checkpoint['epoch']} with IoU {checkpoint['iou']:.4f}")

    # Load dataset using stats from checkpoint
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    dataset = SolarFarmDataset(patches_path, crop_size=CROP_SIZE, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Evaluating on {len(dataset)} samples from {patches_path}")

    # Setup output directory if exporting
    if export_viz:
        if output_dir is None:
            output_dir = "viz"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Exporting visualizations to {output_path}")

    # Evaluate with global metric accumulation
    total_intersection = 0.0
    total_pred_sum = 0.0
    total_target_sum = 0.0
    exported_count = 0

    with torch.no_grad():
        for idx, (embeddings, masks) in enumerate(tqdm(loader, desc="Evaluating")):
            embeddings = embeddings.to(device)
            masks = masks.to(device)

            outputs = model(embeddings)
            outputs = torch.sigmoid(outputs)

            # Accumulate global metrics
            pred_binary = (outputs > 0.5).float()
            total_intersection += (pred_binary * masks).sum().item()
            total_pred_sum += pred_binary.sum().item()
            total_target_sum += masks.sum().item()

            # Export visualization if requested
            if export_viz:
                pred = outputs.squeeze().cpu().numpy()
                mask = masks.squeeze().cpu().numpy()
                pred_bin = (pred > 0.5).astype(np.uint8)
                mask_bin = (mask > 0.5).astype(np.uint8)

                # Skip patches with no positives
                if mask_bin.sum() == 0 and pred_bin.sum() == 0:
                    continue

                # Get geo-referencing from mask TIF
                emb_path = dataset.patch_files[idx]
                mask_tif_path = emb_path.parent / emb_path.name.replace("_emb.npy", "_mask.tif")

                if not mask_tif_path.exists():
                    continue

                with rasterio.open(mask_tif_path) as src:
                    crs = src.crs
                    transform = src.transform

                patch_name = emb_path.stem.replace("_emb", "")

                # Create RGB: Green=TP, Red=FN, Blue=FP
                tp = ((pred_bin == 1) & (mask_bin == 1)).astype(np.uint8) * 255
                fn = ((pred_bin == 0) & (mask_bin == 1)).astype(np.uint8) * 255
                fp = ((pred_bin == 1) & (mask_bin == 0)).astype(np.uint8) * 255
                rgb = np.stack([fn, tp, fp])

                viz_path = output_path / f"{patch_name}_viz.tif"
                with rasterio.open(
                    viz_path,
                    "w",
                    driver="GTiff",
                    height=rgb.shape[1],
                    width=rgb.shape[2],
                    count=3,
                    dtype=np.uint8,
                    crs=crs,
                    transform=transform,
                ) as dst:
                    dst.write(rgb)

                exported_count += 1

    # Compute global metrics
    eps = 1e-6
    total_union = total_pred_sum + total_target_sum - total_intersection
    metrics = {
        "iou": (total_intersection + eps) / (total_union + eps),
        "dice": (2 * total_intersection + eps) / (total_pred_sum + total_target_sum + eps),
        "precision": (total_intersection + eps) / (total_pred_sum + eps),
        "recall": (total_intersection + eps) / (total_target_sum + eps),
    }

    # Print results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Dice:      {metrics['dice']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("=" * 40)

    if export_viz:
        print(f"\nExported {exported_count} visualizations to {output_path}")
        print("  Green = True Positives, Red = False Negatives, Blue = False Positives")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate solar farm segmentation model")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--export-viz", action="store_true",
                        help="Export visualization GeoTIFFs for patches with positives")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for visualizations (default: tmp/)")
    args = parser.parse_args()

    patches_dir = f"patches/{args.split}"
    output_dir = args.output_dir or "tmp"

    evaluate(patches_dir, output_dir, args.export_viz)
