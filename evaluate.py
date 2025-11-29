#!/usr/bin/env python3
"""
Evaluate the trained model on test patches.

This script loads a trained model and evaluates it on a test set,
reporting IoU, Dice, Precision, Recall, and F1 scores.

Usage:
    python evaluate.py --patches patches/test --model models/solar_unet.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNetSmall, SolarFarmDataset, compute_metrics


def main(patches_dir: str, model_path: str):
    """Evaluate model on test set."""
    patches_path = Path(patches_dir)
    model_file = Path(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_file}...")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    model = UNetSmall(in_channels=128, out_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model trained to epoch {checkpoint['epoch']} with IoU {checkpoint['iou']:.4f}")

    # Load test dataset
    test_dataset = SolarFarmDataset(patches_path, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    all_metrics = []

    with torch.no_grad():
        for embeddings, masks in tqdm(test_loader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            masks = masks.to(device)

            outputs = model(embeddings)
            metrics = compute_metrics(outputs, masks)
            all_metrics.append(metrics)

    # Aggregate metrics
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}

    # Print results
    print("\n" + "=" * 40)
    print("TEST RESULTS")
    print("=" * 40)
    print(f"IoU:       {avg_metrics['iou']:.4f}")
    print(f"Dice:      {avg_metrics['dice']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall:    {avg_metrics['recall']:.4f}")
    print(f"F1:        {avg_metrics['f1']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--patches", default="patches/test", help="Test patches directory")
    parser.add_argument("--model", default="models/solar_unet.pth", help="Model path")
    args = parser.parse_args()

    main(args.patches, args.model)
