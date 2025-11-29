#!/usr/bin/env python3
"""
Predict solar farms in a geographic region.

This script fetches a GeoTessera mosaic for a bounding box and runs
sliding window inference to detect solar farms.

Usage:
    python predict.py --bbox "-0.5,51.8,0.0,52.1" --model models/solar_unet.pth
    python predict.py --bbox "-0.5,51.8,0.0,52.1" --model models/solar_unet.pth --output outputs/prediction.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from geotessera import GeoTessera
from models import UNetSmall

# Settings
PATCH_SIZE = 64
STRIDE = 32
YEAR = 2024
THRESHOLD = 0.5


def predict_mosaic(model, mosaic: np.ndarray, device) -> np.ndarray:
    """Run sliding window inference on a mosaic."""
    h, w, _ = mosaic.shape

    prediction = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    model.eval()

    with torch.no_grad():
        for y in tqdm(range(0, h - PATCH_SIZE + 1, STRIDE), desc="Predicting"):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch = mosaic[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]

                # Skip patches with NaN values
                if np.any(np.isnan(patch)):
                    continue

                # Normalize patch
                patch_mean = patch.mean(axis=(0, 1), keepdims=True)
                patch_std = patch.std(axis=(0, 1), keepdims=True) + 1e-8
                patch_norm = (patch - patch_mean) / patch_std

                # Convert to tensor and predict
                patch_tensor = torch.from_numpy(patch_norm).float().permute(2, 0, 1).unsqueeze(0)
                patch_tensor = patch_tensor.to(device)

                output = model(patch_tensor)
                pred = output[0, 0].cpu().numpy()

                # Accumulate predictions
                prediction[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += pred
                count[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

    # Average overlapping predictions
    prediction = np.divide(prediction, count, where=count > 0)
    return prediction


def main(bbox_str: str, model_path: str, output_path: str):
    """Predict solar farms in a region."""
    # Parse bounding box
    try:
        coords = [float(x) for x in bbox_str.split(",")]
        if len(coords) != 4:
            raise ValueError
        min_lon, min_lat, max_lon, max_lat = coords
        bbox = (min_lon, min_lat, max_lon, max_lat)
    except Exception:
        print("Invalid bbox format. Expected 'min_lon,min_lat,max_lon,max_lat'")
        return

    print(f"Bounding box: {bbox}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_file = Path(model_path)
    print(f"Loading model from {model_file}...")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    model = UNetSmall(in_channels=128, out_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Fetch mosaic
    print("Fetching GeoTessera mosaic...")
    gt = GeoTessera()

    try:
        mosaic, _, _ = gt.fetch_mosaic_for_region(
            bbox=bbox, year=YEAR, target_crs="EPSG:4326", auto_download=True
        )
    except Exception as e:
        print(f"Failed to fetch mosaic: {e}")
        return

    print(f"Mosaic shape: {mosaic.shape}")

    # Run prediction
    prediction = predict_mosaic(model, mosaic, device)

    # Save visualization
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(prediction, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(label="Solar farm probability")
    plt.title(f"Prediction for {bbox}")
    plt.axis("off")
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Prediction saved to: {output_file}")

    # Statistics
    binary_pred = prediction > THRESHOLD
    num_positive = binary_pred.sum()
    area_m2 = num_positive * 100  # 10m resolution -> 100 m^2 per pixel

    print(f"\nPrediction statistics (threshold={THRESHOLD}):")
    print(f"  Positive pixels: {num_positive}")
    print(f"  Estimated area: {area_m2:.0f} m^2 ({area_m2 / 10000:.2f} hectares)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict solar farms in a region")
    parser.add_argument("--bbox", required=True, help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--model", default="models/solar_unet.pth", help="Model path")
    parser.add_argument("--output", default="outputs/prediction.png", help="Output image path")
    args = parser.parse_args()

    main(args.bbox, args.model, args.output)
