#!/usr/bin/env python3
"""
Create demo UMAP clustering data for testing the visualization.

This generates synthetic data that mimics real solar farm pixel clustering,
allowing you to test the animation without needing a trained model.
"""

import json
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("animation_data")
N_SAMPLES = 2000
RANDOM_SEED = 42

def generate_demo_data():
    """Generate synthetic pixel data with realistic clustering patterns."""
    np.random.seed(RANDOM_SEED)

    # Create two main clusters in UMAP space
    # Cluster 1: Solar farms (tight cluster)
    n_solar = 300
    solar_umap_center = np.array([0.75, 0.75])
    solar_umap = np.random.randn(n_solar, 2) * 0.08 + solar_umap_center

    # Cluster 2: Non-solar (more spread out)
    n_non_solar = N_SAMPLES - n_solar

    # Create several sub-clusters for non-solar (different land types)
    # Distribute samples across subclusters
    samples_per_subcluster = n_non_solar // 4
    remainder = n_non_solar % 4

    subclusters = [
        (0.2, 0.3, 0.12, samples_per_subcluster + (1 if 0 < remainder else 0)),  # Forest
        (0.4, 0.6, 0.10, samples_per_subcluster + (1 if 1 < remainder else 0)),  # Agricultural
        (0.6, 0.2, 0.08, samples_per_subcluster + (1 if 2 < remainder else 0)),  # Urban
        (0.3, 0.8, 0.09, samples_per_subcluster),  # Water
    ]

    non_solar_umap = []
    for cx, cy, std, n in subclusters:
        cluster = np.random.randn(n, 2) * std + np.array([cx, cy])
        non_solar_umap.append(cluster)

    non_solar_umap = np.vstack(non_solar_umap)

    # Clip to [0, 1] range
    solar_umap = np.clip(solar_umap, 0, 1)
    non_solar_umap = np.clip(non_solar_umap, 0, 1)

    # Generate spatial positions (geographic)
    # Solar farms tend to be in open areas - create some spatial patterns
    mosaic_width, mosaic_height = 800, 600

    # Solar farms: clustered in a few regions
    solar_spatial = []
    solar_regions = [
        (600, 200, 80),  # Region 1
        (300, 400, 60),  # Region 2
        (500, 450, 50),  # Region 3
    ]

    samples_per_region = n_solar // len(solar_regions)
    for cx, cy, std in solar_regions:
        region_samples = np.random.randn(samples_per_region, 2) * std + np.array([cx, cy])
        solar_spatial.append(region_samples)

    solar_spatial = np.vstack(solar_spatial)
    if len(solar_spatial) < n_solar:
        # Add remaining samples
        extra = n_solar - len(solar_spatial)
        solar_spatial = np.vstack([
            solar_spatial,
            np.random.randn(extra, 2) * 60 + np.array([solar_regions[0][0], solar_regions[0][1]])
        ])

    # Non-solar: more evenly distributed
    non_solar_spatial = np.random.rand(n_non_solar, 2) * np.array([mosaic_width, mosaic_height])

    # Clip spatial coordinates
    solar_spatial = np.clip(solar_spatial, 0, [mosaic_width, mosaic_height])

    # Combine all data
    all_umap = np.vstack([solar_umap, non_solar_umap])
    all_spatial = np.vstack([solar_spatial, non_solar_spatial])
    is_solar = np.array([True] * n_solar + [False] * n_non_solar)

    # Generate predictions (with some noise)
    predictions = np.where(
        is_solar,
        np.random.beta(8, 2, N_SAMPLES),  # Solar: high predictions
        np.random.beta(2, 8, N_SAMPLES)   # Non-solar: low predictions
    )

    # Create points list
    points = []
    for i in range(N_SAMPLES):
        points.append({
            'id': i,
            'x': float(all_spatial[i, 0]),
            'y': float(all_spatial[i, 1]),
            'umap_x': float(all_umap[i, 0]),
            'umap_y': float(all_umap[i, 1]),
            'prediction': float(predictions[i]),
            'is_solar': bool(is_solar[i])
        })

    # Shuffle points for better animation
    np.random.shuffle(points)

    output_data = {
        'bbox': [0.4, 51.2, 0.7, 51.4],  # Example Cambridge region
        'mosaic_shape': [mosaic_height, mosaic_width],
        'n_samples': N_SAMPLES,
        'n_solar': int(n_solar),
        'n_non_solar': int(n_non_solar),
        'threshold': 0.5,
        'points': points
    }

    return output_data


def main():
    print("=" * 70)
    print("Demo UMAP Clustering Data Generator")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n[1/2] Generating synthetic pixel data ({N_SAMPLES} samples)...")
    data = generate_demo_data()

    print(f"[2/2] Exporting data...")
    output_file = OUTPUT_DIR / "umap_clustering.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Demo data exported to {output_file}")
    print(f"  - Solar farm pixels: {data['n_solar']}")
    print(f"  - Non-solar pixels: {data['n_non_solar']}")
    print(f"  - Total samples: {data['n_samples']}")

    print("\n" + "=" * 70)
    print("Demo data ready!")
    print("\nNext steps:")
    print("1. cd animation_data")
    print("2. python -m http.server 8000")
    print("3. Open http://localhost:8000 in your browser")
    print("=" * 70)


if __name__ == "__main__":
    main()
