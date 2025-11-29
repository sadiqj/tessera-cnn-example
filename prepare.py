#!/usr/bin/env python3
"""
Prepare training patches from GeoTessera embeddings.

This script extracts patches from satellite imagery embeddings around known
solar farm locations. Each patch includes:
- A 64x64 embedding array (128 channels from GeoTessera)
- A binary mask indicating solar farm pixels

Usage:
    python prepare.py --input data/train_solar_farms.geojson --output patches/train
    python prepare.py --input data/test_solar_farms.geojson --output patches/test
"""

import argparse
import json
from pathlib import Path

import geojson
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from rasterio import features
from shapely.geometry import shape
from shapely.ops import transform
from tqdm import tqdm

from geotessera import GeoTessera

# Settings
PATCH_SIZE = 64
PATCH_STRIDE = 32
YEAR = 2024
MIN_POSITIVE_PIXELS = 10


def load_polygons(geojson_path: Path) -> list:
    """Load solar farm polygons from a GeoJSON file."""
    with open(geojson_path) as f:
        data = geojson.load(f)

    polygons = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        polygons.append({
            "geometry": geom,
            "bounds": geom.bounds,  # (minx, miny, maxx, maxy)
        })

    print(f"Loaded {len(polygons)} polygons from {geojson_path}")
    return polygons


def transform_polygon(polygon, src_crs: str, dst_crs: str):
    """Transform a polygon between coordinate systems."""
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, polygon)


def extract_patches_for_polygon(gt: GeoTessera, polygon_data: dict) -> list:
    """Extract patches from tiles covering a polygon."""
    patches = []
    geom = polygon_data["geometry"]
    minx, miny, maxx, maxy = geom.bounds

    # Add buffer in degrees (approximately 100m)
    lat_center = (miny + maxy) / 2
    buffer_deg_lon = 100.0 / (111320 * np.cos(np.radians(lat_center)))
    buffer_deg_lat = 100.0 / 111320
    bbox = (minx - buffer_deg_lon, miny - buffer_deg_lat, maxx + buffer_deg_lon, maxy + buffer_deg_lat)

    # Get tiles covering this area
    tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=YEAR)
    if not tiles:
        return patches

    for year, tile_lon, tile_lat in tiles:
        try:
            embedding, crs, tile_transform = gt.fetch_embedding(tile_lon, tile_lat, year)
        except Exception:
            continue

        h, w, _ = embedding.shape

        # Rasterize polygon to tile coordinates
        polygon_transformed = transform_polygon(geom, "EPSG:4326", str(crs))
        mask = features.rasterize(
            [(polygon_transformed, 1)],
            out_shape=(h, w),
            transform=tile_transform,
            fill=0,
            dtype=np.uint8,
        )

        # Extract patches with sliding window
        for y in range(0, h - PATCH_SIZE + 1, PATCH_STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_STRIDE):
                emb_patch = embedding[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]
                mask_patch = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                # Skip patches with NaN values
                if np.any(np.isnan(emb_patch)):
                    continue

                # Only keep patches with some positive pixels
                if mask_patch.sum() >= MIN_POSITIVE_PIXELS:
                    patches.append((emb_patch.copy(), mask_patch.copy()))

    return patches


def extract_negative_patches(gt: GeoTessera, polygon_list: list, num_patches: int) -> list:
    """Sample negative patches from tiles adjacent to solar farms."""
    patches = []

    # Get all tiles containing polygons
    positive_tiles = set()
    for poly_data in polygon_list:
        tiles = gt.registry.load_blocks_for_region(bounds=poly_data["bounds"], year=YEAR)
        for _, lon, lat in tiles:
            positive_tiles.add((lon, lat))

    # Find adjacent tiles that don't contain solar farms
    candidate_tiles = set()
    for lon, lat in positive_tiles:
        for dlon in [-0.1, 0, 0.1]:
            for dlat in [-0.1, 0, 0.1]:
                if dlon == 0 and dlat == 0:
                    continue
                neighbor = (round(lon + dlon, 2), round(lat + dlat, 2))
                if neighbor not in positive_tiles:
                    candidate_tiles.add(neighbor)

    if not candidate_tiles:
        return patches

    # Sample from candidate tiles
    candidate_list = list(candidate_tiles)
    np.random.shuffle(candidate_list)

    for tile_lon, tile_lat in candidate_list:
        if len(patches) >= num_patches:
            break

        try:
            embedding, _, _ = gt.fetch_embedding(tile_lon, tile_lat, YEAR)
        except Exception:
            continue

        h, w, _ = embedding.shape
        empty_mask = np.zeros((h, w), dtype=np.uint8)

        # Extract a few patches from this tile
        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                if len(patches) >= num_patches:
                    break

                emb_patch = embedding[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]
                if np.any(np.isnan(emb_patch)):
                    continue

                mask_patch = empty_mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                patches.append((emb_patch.copy(), mask_patch.copy()))

    return patches


def main(input_geojson: str, output_dir: str):
    """Prepare patches from GeoJSON polygons."""
    input_path = Path(input_geojson)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gt = GeoTessera()
    polygons = load_polygons(input_path)

    # Extract positive patches
    all_patches = []
    print("Extracting patches around solar farms...")
    for poly_data in tqdm(polygons, desc="Processing polygons"):
        patches = extract_patches_for_polygon(gt, poly_data)
        all_patches.extend(patches)

    positive_count = len(all_patches)
    print(f"Extracted {positive_count} positive patches")

    # Extract negative patches (roughly equal to positive)
    print("Extracting negative patches...")
    negative_patches = extract_negative_patches(gt, polygons, positive_count)
    all_patches.extend(negative_patches)

    negative_count = len(negative_patches)
    print(f"Extracted {negative_count} negative patches")

    # Save patches
    print(f"Saving {len(all_patches)} patches to {output_path}...")
    for i, (emb, mask) in enumerate(tqdm(all_patches, desc="Saving")):
        np.save(output_path / f"patch_{i:06d}_emb.npy", emb.astype(np.float32))
        np.save(output_path / f"patch_{i:06d}_mask.npy", mask.astype(np.uint8))
        plt.imsave(output_path / f"patch_{i:06d}_mask.png", mask, cmap="gray", vmin=0, vmax=1)

    # Save metadata
    metadata = {
        "num_patches": len(all_patches),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "patch_size": PATCH_SIZE,
        "year": YEAR,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Saved {len(all_patches)} patches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training patches from GeoTessera")
    parser.add_argument("--input", default="data/train_solar_farms.geojson", help="Input GeoJSON file")
    parser.add_argument("--output", default="patches/train", help="Output directory")
    args = parser.parse_args()

    main(args.input, args.output)
