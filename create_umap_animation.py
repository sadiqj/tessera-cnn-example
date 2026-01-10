#!/usr/bin/env python3
"""
Create UMAP clustering animation data for solar farm pixels.

This script:
1. Fetches a region around a solar farm
2. Extracts pixel embeddings
3. Samples pixels for visualization
4. Uses the CNN model to predict solar farm pixels
5. Applies UMAP to cluster the embeddings
6. Exports data for anime.js visualization
"""

import json
from pathlib import Path

import numpy as np
import torch
from geotessera import GeoTessera
from models import UNetSmall

# Configuration
BBOX = (0.4, 51.2, 0.7, 51.4)  # Area with solar farms (Cambridge/Suffolk region)
MODEL_PATH = "models/solar_unet.pth"
OUTPUT_DIR = Path("animation_data")
YEAR = 2024
SAMPLE_SIZE = 2000  # Number of pixels to sample for visualization
RANDOM_SEED = 42

# Solar farm threshold
THRESHOLD = 0.5


def sample_pixels(mosaic: np.ndarray, prediction: np.ndarray, n_samples: int, seed: int = 42) -> dict:
    """
    Sample pixels from the mosaic for UMAP clustering.

    Returns a dict with:
    - embeddings: (n_samples, 128) array
    - predictions: (n_samples,) array of probabilities
    - positions: (n_samples, 2) array of x,y positions
    - is_solar: (n_samples,) boolean array
    """
    np.random.seed(seed)

    h, w, c = mosaic.shape

    # Create valid pixel mask (no NaN values)
    valid_mask = ~np.any(np.isnan(mosaic), axis=2)
    valid_indices = np.argwhere(valid_mask)

    # Sample pixels
    if len(valid_indices) > n_samples:
        sample_idx = np.random.choice(len(valid_indices), n_samples, replace=False)
        sampled_positions = valid_indices[sample_idx]
    else:
        sampled_positions = valid_indices
        n_samples = len(valid_indices)

    # Extract data
    embeddings = np.zeros((n_samples, c), dtype=np.float32)
    predictions = np.zeros(n_samples, dtype=np.float32)
    positions = np.zeros((n_samples, 2), dtype=np.float32)

    for i, (y, x) in enumerate(sampled_positions):
        embeddings[i] = mosaic[y, x]
        predictions[i] = prediction[y, x]
        positions[i] = [x, y]

    is_solar = predictions > THRESHOLD

    return {
        'embeddings': embeddings,
        'predictions': predictions,
        'positions': positions,
        'is_solar': is_solar
    }


def predict_region(model, mosaic: np.ndarray, device, mean: np.ndarray, std: np.ndarray,
                   patch_size: int = 64) -> np.ndarray:
    """Run inference on the entire mosaic."""
    h, w, _ = mosaic.shape
    prediction = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    model.eval()

    # Use stride equal to patch size for faster inference
    stride = patch_size

    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = mosaic[y:y + patch_size, x:x + patch_size, :]

                # Skip patches with NaN values
                if np.any(np.isnan(patch)):
                    continue

                # Normalize
                patch_norm = (patch - mean) / std

                # Convert to tensor and predict
                patch_tensor = torch.from_numpy(patch_norm).float().permute(2, 0, 1).unsqueeze(0)
                patch_tensor = patch_tensor.to(device)

                output = model(patch_tensor)
                output = torch.sigmoid(output)
                pred = output[0, 0].cpu().numpy()

                # Accumulate predictions
                prediction[y:y + patch_size, x:x + patch_size] += pred
                count[y:y + patch_size, x:x + patch_size] += 1

        # Handle edges with overlapping patches
        if h % stride != 0 or w % stride != 0:
            # Add edge patches
            if h % stride != 0:
                y = h - patch_size
                for x in range(0, w - patch_size + 1, stride):
                    patch = mosaic[y:y + patch_size, x:x + patch_size, :]
                    if not np.any(np.isnan(patch)):
                        patch_norm = (patch - mean) / std
                        patch_tensor = torch.from_numpy(patch_norm).float().permute(2, 0, 1).unsqueeze(0)
                        patch_tensor = patch_tensor.to(device)
                        output = torch.sigmoid(model(patch_tensor))
                        pred = output[0, 0].cpu().numpy()
                        prediction[y:y + patch_size, x:x + patch_size] += pred
                        count[y:y + patch_size, x:x + patch_size] += 1

            if w % stride != 0:
                x = w - patch_size
                for y in range(0, h - patch_size + 1, stride):
                    patch = mosaic[y:y + patch_size, x:x + patch_size, :]
                    if not np.any(np.isnan(patch)):
                        patch_norm = (patch - mean) / std
                        patch_tensor = torch.from_numpy(patch_norm).float().permute(2, 0, 1).unsqueeze(0)
                        patch_tensor = patch_tensor.to(device)
                        output = torch.sigmoid(model(patch_tensor))
                        pred = output[0, 0].cpu().numpy()
                        prediction[y:y + patch_size, x:x + patch_size] += pred
                        count[y:y + patch_size, x:x + patch_size] += 1

    # Average overlapping predictions
    np.divide(prediction, count, where=count > 0, out=prediction)
    return prediction


def apply_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Apply UMAP dimensionality reduction."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Installing umap-learn...")
        import subprocess
        subprocess.check_call(["pip", "install", "umap-learn"])
        import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=RANDOM_SEED,
        verbose=False
    )

    embedding_2d = reducer.fit_transform(embeddings)
    return embedding_2d


def main():
    print("=" * 70)
    print("UMAP Clustering Animation Generator")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/7] Using device: {device}")

    # Load model
    print(f"[2/7] Loading model from {MODEL_PATH}...")
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train a model first using train.py")
        return

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)

    model = UNetSmall(in_channels=128, out_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print("  Model loaded successfully!")

    # Fetch mosaic
    print(f"[3/7] Fetching GeoTessera mosaic for {BBOX}...")
    gt = GeoTessera()

    try:
        mosaic, crs, mosaic_transform = gt.fetch_mosaic_for_region(
            bbox=BBOX, year=YEAR, target_crs="EPSG:4326", auto_download=True
        )
    except Exception as e:
        print(f"Error fetching mosaic: {e}")
        return

    print(f"  Mosaic shape: {mosaic.shape}")

    # Run prediction
    print("[4/7] Running CNN inference on the region...")
    prediction = predict_region(model, mosaic, device, mean, std)

    # Count detected solar farms
    solar_pixels = (prediction > THRESHOLD).sum()
    total_pixels = prediction.size
    print(f"  Detected {solar_pixels}/{total_pixels} solar farm pixels ({100*solar_pixels/total_pixels:.2f}%)")

    # Sample pixels
    print(f"[5/7] Sampling {SAMPLE_SIZE} pixels for visualization...")
    sample_data = sample_pixels(mosaic, prediction, SAMPLE_SIZE, RANDOM_SEED)

    n_solar = sample_data['is_solar'].sum()
    n_non_solar = len(sample_data['is_solar']) - n_solar
    print(f"  Sampled {n_solar} solar farm pixels and {n_non_solar} non-solar pixels")

    # Apply UMAP
    print("[6/7] Applying UMAP clustering...")
    umap_coords = apply_umap(sample_data['embeddings'])
    print("  UMAP clustering complete!")

    # Normalize UMAP coordinates to [0, 1] range for easier visualization
    umap_min = umap_coords.min(axis=0)
    umap_max = umap_coords.max(axis=0)
    umap_normalized = (umap_coords - umap_min) / (umap_max - umap_min)

    # Prepare data for export
    print("[7/7] Exporting data for visualization...")

    # Convert to serializable format
    points = []
    for i in range(len(sample_data['is_solar'])):
        points.append({
            'id': i,
            'x': float(sample_data['positions'][i, 0]),
            'y': float(sample_data['positions'][i, 1]),
            'umap_x': float(umap_normalized[i, 0]),
            'umap_y': float(umap_normalized[i, 1]),
            'prediction': float(sample_data['predictions'][i]),
            'is_solar': bool(sample_data['is_solar'][i])
        })

    output_data = {
        'bbox': BBOX,
        'mosaic_shape': mosaic.shape[:2],
        'n_samples': len(points),
        'n_solar': int(n_solar),
        'n_non_solar': int(n_non_solar),
        'threshold': THRESHOLD,
        'points': points
    }

    # Save to JSON
    output_file = OUTPUT_DIR / "umap_clustering.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Data exported to {output_file}")
    print(f"  Total points: {len(points)}")

    print("\n" + "=" * 70)
    print("Data generation complete!")
    print("Next step: Open animation_data/index.html in a browser")
    print("=" * 70)


if __name__ == "__main__":
    main()
