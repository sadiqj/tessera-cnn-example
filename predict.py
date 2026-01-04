#!/usr/bin/env python3
"""
Predict solar farms in a geographic region.

This script fetches a GeoTessera mosaic for a bounding box and runs
sliding window inference to detect solar farms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from tqdm import tqdm

from geotessera import GeoTessera
from models import UNet

# Settings
BBOX = (-0.5, 51.8, 0.0, 52.1)  # (min_lon, min_lat, max_lon, max_lat)
MODEL_PATH = "models/solar_unet.pth"
OUTPUT_PATH = "outputs/prediction.png"
PATCH_SIZE = 64
STRIDE = 32
YEAR = 2024
THRESHOLD = 0.5


def predict_mosaic(model, mosaic: np.ndarray, device, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Run sliding window inference on a mosaic using global normalization stats."""
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

                # Normalize using global stats from training set
                patch_norm = (patch - mean) / std

                # Convert to tensor and predict
                patch_tensor = torch.from_numpy(patch_norm).float().permute(2, 0, 1).unsqueeze(0)
                patch_tensor = patch_tensor.to(device)

                output = model(patch_tensor)
                output = torch.sigmoid(output)
                pred = output[0, 0].cpu().numpy()

                # Accumulate predictions
                prediction[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += pred
                count[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

    # Average overlapping predictions
    np.divide(prediction, count, where=count > 0, out=prediction)
    return prediction


def main():
    """Predict solar farms in a region."""
    print(f"Bounding box: {BBOX}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and extract normalization stats from checkpoint
    model_file = Path(MODEL_PATH)
    print(f"Loading model from {model_file}...")
    model, checkpoint = UNet.from_checkpoint(str(model_file), device=str(device))
    model = model.to(device)

    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)

    # Fetch mosaic
    print("Fetching GeoTessera mosaic...")
    gt = GeoTessera()

    try:
        mosaic, mosaic_transform, crs = gt.fetch_mosaic_for_region(
            bbox=BBOX, year=YEAR, target_crs="EPSG:4326", auto_download=True
        )
    except Exception as e:
        print(f"Failed to fetch mosaic: {e}")
        return

    print(f"Mosaic shape: {mosaic.shape}")

    # Run prediction
    prediction = predict_mosaic(model, mosaic, device, mean, std)

    # Save visualization
    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(prediction, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(label="Solar farm probability")
    plt.title(f"Prediction for {BBOX}")
    plt.axis("off")
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Prediction saved to: {output_file}")

    # Save georeferenced GeoTIFF
    tif_file = output_file.parent / "prediction.tif"
    with rasterio.open(
        tif_file,
        "w",
        driver="GTiff",
        height=prediction.shape[0],
        width=prediction.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=mosaic_transform,
    ) as dst:
        dst.write(prediction, 1)

    print(f"GeoTIFF saved to: {tif_file}")

    # Statistics
    binary_pred = prediction > THRESHOLD
    num_positive = binary_pred.sum()
    area_m2 = num_positive * 100  # 10m resolution -> 100 m^2 per pixel

    print(f"\nPrediction statistics (threshold={THRESHOLD}):")
    print(f"  Positive pixels: {num_positive}")
    print(f"  Estimated area: {area_m2:.0f} m^2 ({area_m2 / 10000:.2f} hectares)")


if __name__ == "__main__":
    main()
