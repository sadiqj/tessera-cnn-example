#!/usr/bin/env python3
"""
UK-wide solar farm inference pipeline.

This script runs inference over all GeoTessera tiles covering the UK,
converts detections to GeoJSON polygons, and merges them into a final output.

Features:
- Processes one tile at a time to manage memory
- Resumable: tracks completed tiles, skips on restart
- Saves per-tile probability GeoTIFFs and GeoJSON polygons
- Merges touching polygons across tile boundaries using buffer/dissolve

Usage:
    uv run predict_uk.py --model models/solar_unet.pth --year 2024 --output-dir outputs/uk_solar_2024
"""

import argparse
import json
import logging
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import torch
from rasterio.features import rasterize, shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from tqdm import tqdm

from geotessera import GeoTessera
from models import UNet

# UK bounding box (includes sea for simplicity)
UK_BBOX = (-8.5, 49.0, 2.0, 61.0)  # (min_lon, min_lat, max_lon, max_lat)

# Inference settings
PATCH_SIZE = 64
STRIDE = 32
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_AREA_M2 = 10000  # 1 hectare
DEFAULT_BUFFER_M = 20.0  # Merge polygons within 2 pixels of each other
DEFAULT_MIN_CONFIDENCE = 0.9  # Only include solar farms where we're 90%+ sure on average

# URL for UK boundary (Natural Earth admin 0 countries)
UK_BOUNDARY_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
UK_COUNTRY_NAMES = ["United Kingdom", "Isle of Man"]  # Include Crown Dependencies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Progress:
    """Track processing progress for resumability."""

    completed_tiles: set = field(default_factory=set)
    failed_tiles: dict = field(default_factory=dict)

    @staticmethod
    def tile_key(lon: float, lat: float) -> str:
        return f"{lon:.2f}_{lat:.2f}"

    def is_done(self, lon: float, lat: float) -> bool:
        return self.tile_key(lon, lat) in self.completed_tiles

    def mark_done(self, lon: float, lat: float):
        self.completed_tiles.add(self.tile_key(lon, lat))

    def mark_failed(self, lon: float, lat: float, error: str):
        self.failed_tiles[self.tile_key(lon, lat)] = error

    @classmethod
    def load(cls, path: Path) -> "Progress":
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(
                completed_tiles=set(data.get("completed", [])),
                failed_tiles=data.get("failed", {}),
            )
        return cls()

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(
                {
                    "completed": list(self.completed_tiles),
                    "failed": self.failed_tiles,
                },
                f,
                indent=2,
            )


def predict_tile(
    model: torch.nn.Module,
    embedding: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run sliding window inference on a single tile embedding.

    Args:
        model: Loaded UNet model
        embedding: Tile embedding array (H, W, 128)
        mean, std: Normalization statistics from checkpoint
        device: PyTorch device

    Returns:
        Probability array (H, W) in range [0, 1]
    """
    h, w, _ = embedding.shape
    prediction = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch = embedding[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]

                if np.any(np.isnan(patch)):
                    continue

                patch_norm = (patch - mean) / std
                patch_tensor = torch.from_numpy(patch_norm).float()
                patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

                output = model(patch_tensor)
                output = torch.sigmoid(output)
                pred = output[0, 0].cpu().numpy()

                prediction[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += pred
                count[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

    np.divide(prediction, count, where=count > 0, out=prediction)
    return prediction


def save_probability_geotiff(
    prediction: np.ndarray,
    crs,
    transform,
    output_path: Path,
):
    """Save probability map as georeferenced GeoTIFF with LZW compression."""
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=prediction.shape[0],
        width=prediction.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(prediction, 1)


def polygonize_predictions(
    prediction: np.ndarray,
    crs,
    transform,
    threshold: float,
    min_area_m2: float,
    min_confidence: float,
    year: int,
    tile_key: str,
) -> list:
    """Convert probability raster to polygons with properties.

    Args:
        prediction: Probability array (H, W)
        crs, transform: Georeferencing info
        threshold: Minimum probability for inclusion
        min_area_m2: Minimum polygon area
        min_confidence: Minimum average confidence for a polygon to be included
        year: Imagery year for properties
        tile_key: Tile identifier for debugging

    Returns:
        List of GeoJSON feature dicts
    """
    binary = (prediction > threshold).astype(np.uint8)

    if binary.sum() == 0:
        return []

    features_list = []

    for geom, value in shapes(binary, mask=binary, transform=transform):
        if value == 0:
            continue

        poly = shape(geom)

        # Calculate area in the tile's CRS
        # Note: CRS should be projected (meters) for accurate area
        # For tiles in EPSG:4326, we need to reproject for area calculation
        crs_str = str(crs)
        if "4326" in crs_str or crs.is_geographic:
            # Reproject to British National Grid for area calculation
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:27700", always_xy=True)
            poly_projected = shapely_transform(transformer.transform, poly)
            area_m2 = poly_projected.area
        else:
            area_m2 = poly.area

        if area_m2 < min_area_m2:
            continue

        # Calculate mean confidence within polygon
        poly_mask = rasterize(
            [(geom, 1)],
            out_shape=prediction.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        mean_conf = float(prediction[poly_mask == 1].mean())

        # Skip polygons below minimum confidence threshold
        if mean_conf < min_confidence:
            continue

        # Transform polygon and centroid to WGS84
        if "4326" not in crs_str and not crs.is_geographic:
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            poly_4326 = shapely_transform(transformer.transform, poly)
        else:
            poly_4326 = poly

        centroid = poly_4326.centroid

        features_list.append(
            {
                "type": "Feature",
                "geometry": mapping(poly_4326),
                "properties": {
                    "area_m2": round(area_m2, 1),
                    "centroid": [round(centroid.x, 6), round(centroid.y, 6)],
                    "mean_confidence": round(mean_conf, 4),
                    "year": year,
                    "tile": tile_key,
                },
            }
        )

    return features_list


def process_single_tile(
    gt: GeoTessera,
    model: torch.nn.Module,
    year: int,
    tile_lon: float,
    tile_lat: float,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    min_area_m2: float,
    min_confidence: float,
) -> tuple[bool, str]:
    """Process a single tile: fetch, predict, polygonize, save.

    Returns:
        (success, error_message) tuple
    """
    tile_key = f"tile_{tile_lon:.2f}_{tile_lat:.2f}"
    tile_dir = output_dir / "tiles" / tile_key
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed (fallback for progress file issues)
    prob_path = tile_dir / "probability.tif"
    geojson_path = tile_dir / "polygons.geojson"
    if prob_path.exists() and geojson_path.exists():
        return True, ""

    try:
        embedding, crs, tile_transform = gt.fetch_embedding(tile_lon, tile_lat, year)
    except Exception as e:
        return False, f"Failed to fetch embedding: {e}"

    try:
        prediction = predict_tile(model, embedding, mean, std, device)
    except Exception as e:
        return False, f"Inference failed: {e}"

    try:
        save_probability_geotiff(prediction, crs, tile_transform, prob_path)
    except Exception as e:
        return False, f"Failed to save probability GeoTIFF: {e}"

    try:
        features = polygonize_predictions(
            prediction,
            crs,
            tile_transform,
            threshold=threshold,
            min_area_m2=min_area_m2,
            min_confidence=min_confidence,
            year=year,
            tile_key=tile_key,
        )
    except Exception as e:
        return False, f"Polygonization failed: {e}"

    try:
        fc = {"type": "FeatureCollection", "features": features}
        with open(geojson_path, "w") as f:
            json.dump(fc, f)
    except Exception as e:
        return False, f"Failed to save GeoJSON: {e}"

    return True, ""


def get_uk_land_boundary(cache_dir: Path) -> gpd.GeoDataFrame:
    """Download and cache UK land boundary from Natural Earth.

    Returns a GeoDataFrame with the UK land boundary in EPSG:4326.
    """
    cache_path = cache_dir / "uk_boundary.geojson"

    if cache_path.exists():
        logger.info(f"Loading cached UK boundary from {cache_path}")
        return gpd.read_file(cache_path)

    logger.info("Downloading UK boundary from Natural Earth...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "countries.zip"
        urllib.request.urlretrieve(UK_BOUNDARY_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        shp_path = Path(tmpdir) / "ne_10m_admin_0_countries.shp"
        world = gpd.read_file(shp_path)

        uk = world[world["ADMIN"].isin(UK_COUNTRY_NAMES)]
        uk = uk.dissolve()
        uk = uk[["geometry"]].to_crs("EPSG:4326")

    uk.to_file(cache_path, driver="GeoJSON")
    logger.info(f"Cached UK boundary to {cache_path}")

    return uk


def merge_all_polygons(
    output_dir: Path,
    buffer_m: float,
    min_area_m2: float,
    year: int,
) -> Path | None:
    """Load all per-tile GeoJSONs and merge touching polygons.

    Algorithm:
    1. Load all polygons from tile GeoJSONs
    2. Reproject to British National Grid for meter-based operations
    3. Buffer polygons slightly
    4. Union/dissolve overlapping polygons
    5. Shrink buffer back
    6. Filter out polygons smaller than min_area_m2
    7. Recalculate properties
    8. Save final GeoJSON in WGS84
    """
    tiles_dir = output_dir / "tiles"
    all_features = []

    logger.info("Loading per-tile GeoJSONs...")
    tile_dirs = list(tiles_dir.iterdir()) if tiles_dir.exists() else []

    for tile_dir in tqdm(tile_dirs, desc="Loading tiles"):
        geojson_path = tile_dir / "polygons.geojson"
        if geojson_path.exists():
            with open(geojson_path) as f:
                fc = json.load(f)
            all_features.extend(fc["features"])

    if not all_features:
        logger.warning("No polygons found to merge")
        return None

    logger.info(f"Loaded {len(all_features)} polygons from {len(tile_dirs)} tiles")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

    if gdf.empty:
        logger.warning("GeoDataFrame is empty")
        return None

    # Reproject to British National Grid for meter-based operations
    logger.info("Reprojecting to British National Grid...")
    gdf_bng = gdf.to_crs("EPSG:27700")

    # Buffer, union, negative buffer
    logger.info(f"Buffering polygons by {buffer_m}m and dissolving...")
    buffered = gdf_bng.geometry.buffer(buffer_m)
    merged = unary_union(buffered)

    # Handle both single polygon and multipolygon results
    if merged.is_empty:
        logger.warning("Merged geometry is empty")
        return None

    if merged.geom_type == "Polygon":
        merged_geoms = [merged]
    elif merged.geom_type == "MultiPolygon":
        merged_geoms = list(merged.geoms)
    else:
        merged_geoms = []

    # Shrink back
    logger.info(f"Shrinking buffer back and filtering by area >= {min_area_m2} m²...")
    final_geoms = []
    for geom in merged_geoms:
        shrunk = geom.buffer(-buffer_m)
        if shrunk.is_empty:
            continue

        if shrunk.geom_type == "MultiPolygon":
            for g in shrunk.geoms:
                if g.area >= min_area_m2:
                    final_geoms.append(g)
        elif shrunk.geom_type == "Polygon":
            if shrunk.area >= min_area_m2:
                final_geoms.append(shrunk)

    logger.info(f"After merge: {len(final_geoms)} polygons")

    if not final_geoms:
        logger.warning("No polygons remaining after merge")
        return None

    # Create output GeoDataFrame and convert to WGS84
    merged_gdf = gpd.GeoDataFrame(geometry=final_geoms, crs="EPSG:27700")
    merged_gdf_4326 = merged_gdf.to_crs("EPSG:4326")

    # Clip to UK land boundary
    logger.info("Clipping to UK land boundary...")
    uk_boundary = get_uk_land_boundary(output_dir)
    uk_land = uk_boundary.union_all()

    merged_gdf_4326 = merged_gdf_4326[merged_gdf_4326.intersects(uk_land)]
    merged_gdf_4326 = merged_gdf_4326.clip(uk_land)

    # Calculate area in BNG after clipping
    merged_gdf_bng = merged_gdf_4326.to_crs("EPSG:27700")
    merged_gdf_4326["area_m2"] = merged_gdf_bng.geometry.area.round(1)

    # Filter out polygons that are now too small after clipping
    merged_gdf_4326 = merged_gdf_4326[merged_gdf_4326["area_m2"] >= min_area_m2]

    logger.info(f"After clipping to UK land: {len(merged_gdf_4326)} polygons")

    if merged_gdf_4326.empty:
        logger.warning("No polygons remaining after clipping to UK land")
        return None

    # Calculate centroids in projected CRS, then convert to WGS84 coords
    centroids_bng = merged_gdf_bng.geometry.centroid
    centroids_4326 = centroids_bng.to_crs("EPSG:4326")
    merged_gdf_4326["centroid"] = centroids_4326.apply(
        lambda p: [round(p.x, 6), round(p.y, 6)]
    )
    merged_gdf_4326["year"] = year
    merged_gdf_4326["mean_confidence"] = None  # Would require loading probability rasters

    # Select final columns
    merged_gdf_4326 = merged_gdf_4326[
        ["geometry", "area_m2", "centroid", "year", "mean_confidence"]
    ]

    # Save
    output_path = output_dir / "uk_solar_farms.geojson"
    merged_gdf_4326.to_file(output_path, driver="GeoJSON")

    logger.info(f"Saved merged GeoJSON to {output_path}")
    return output_path


def run_inference(
    model_path: Path,
    output_dir: Path,
    year: int,
    threshold: float,
    min_area_m2: float,
    min_confidence: float,
    buffer_m: float,
    skip_merge: bool,
):
    """Main inference pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {model_path}...")
    model, checkpoint = UNet.from_checkpoint(str(model_path), device=str(device))
    model = model.to(device)
    model.eval()

    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)

    # Initialize GeoTessera
    gt = GeoTessera()

    # Get tiles covering UK
    logger.info(f"Finding tiles covering UK bbox {UK_BBOX} for year {year}...")
    tiles = gt.registry.load_blocks_for_region(bounds=UK_BBOX, year=year)
    logger.info(f"Found {len(tiles)} tiles covering UK")

    # Load progress
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    progress = Progress.load(progress_path)

    # Filter to unprocessed tiles
    tiles_to_process = [
        (y, lon, lat) for y, lon, lat in tiles if not progress.is_done(lon, lat)
    ]
    logger.info(
        f"Tiles to process: {len(tiles_to_process)} "
        f"(already done: {len(progress.completed_tiles)}, "
        f"previously failed: {len(progress.failed_tiles)})"
    )

    # Process tiles
    new_failures = 0
    for tile_year, tile_lon, tile_lat in tqdm(tiles_to_process, desc="Processing UK tiles"):
        success, error = process_single_tile(
            gt=gt,
            model=model,
            year=tile_year,
            tile_lon=tile_lon,
            tile_lat=tile_lat,
            mean=mean,
            std=std,
            device=device,
            output_dir=output_dir,
            threshold=threshold,
            min_area_m2=min_area_m2,
            min_confidence=min_confidence,
        )

        if success:
            progress.mark_done(tile_lon, tile_lat)
        else:
            progress.mark_failed(tile_lon, tile_lat, error)
            new_failures += 1
            logger.warning(f"Tile {tile_lon:.2f},{tile_lat:.2f} failed: {error}")

        progress.save(progress_path)

        # Clear GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    logger.info(
        f"\nProcessing complete. "
        f"Total completed: {len(progress.completed_tiles)}, "
        f"Total failed: {len(progress.failed_tiles)} "
        f"(new failures this run: {new_failures})"
    )

    # Merge step
    if not skip_merge:
        logger.info("Starting merge step...")
        merge_all_polygons(output_dir, buffer_m=buffer_m, min_area_m2=min_area_m2, year=year)
    else:
        logger.info("Skipping merge step (--skip-merge flag set)")


def main():
    parser = argparse.ArgumentParser(
        description="UK-wide solar farm inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Imagery year",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for tiles and merged GeoJSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for polygonization",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=DEFAULT_MIN_AREA_M2,
        help="Minimum polygon area in m^2",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum average confidence for a polygon to be included (0-1)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=DEFAULT_BUFFER_M,
        help="Buffer distance in meters for polygon merge",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip final merge step (useful for debugging or re-running merge separately)",
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        output_dir=args.output_dir,
        year=args.year,
        threshold=args.threshold,
        min_area_m2=args.min_area,
        min_confidence=args.min_confidence,
        buffer_m=args.buffer,
        skip_merge=args.skip_merge,
    )


if __name__ == "__main__":
    main()
