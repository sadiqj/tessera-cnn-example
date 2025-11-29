#!/usr/bin/env python3
"""
Prepare training patches from GeoTessera embeddings.

This script:
1. Downloads solar farm polygons from OpenStreetMap (or uses existing GeoJSON)
2. Filters labels against REPD (Renewable Energy Planning Database) for temporal alignment
3. Splits data by longitude (left-to-right) into train/val/test (70/15/15)
4. Extracts patches from satellite imagery embeddings
5. Creates train/val/test datasets with balanced positive/negative samples
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path

import geojson
import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests
from rasterio import features
from shapely.geometry import shape
from shapely.ops import transform
from tqdm import tqdm

from geotessera import GeoTessera

# Settings
PATCH_SIZE = 160  # Larger patches for random crop augmentation (model uses 128x128)
PATCH_STRIDE = 128
YEAR = 2024
MIN_POSITIVE_PIXELS = 50  # Minimum positive pixels per patch (increased from 10)
MIN_COVERAGE_RATIO = 0.02  # At least 2% of patch must be positive
MAX_COVERAGE_RATIO = 0.95  # At most 95% positive (full coverage is suspicious)

# OSM download settings
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
MIN_AREA_M2 = 10000  # Filter out small installations (< 1 hectare)
CACHE_FILE = "data/osm_solar_farms_cache.json"
REPD_FILE = "data/repd.csv"


def parse_osm_start_date(date_str: str) -> int | None:
    """Parse OSM start_date field to extract year. Returns None if unparseable."""
    if not date_str or pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    # Try various formats
    # "2016-06-01", "2014-03-31"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return int(date_str[:4])

    # "2014", "2015"
    if re.match(r"^\d{4}$", date_str):
        return int(date_str)

    # "Jan 2016", "March 2015"
    match = re.search(r"(\d{4})", date_str)
    if match:
        return int(match.group(1))

    # "Q4 2025"
    match = re.match(r"Q\d\s+(\d{4})", date_str)
    if match:
        return int(match.group(1))

    return None


def load_repd_operational_dates(repd_path: Path) -> dict:
    """Load REPD data and return dict of Old Ref ID -> operational year."""
    if not repd_path.exists():
        print(f"Warning: REPD file not found at {repd_path}")
        return {}

    df = pd.read_csv(repd_path, encoding="latin-1")

    # Filter to solar only
    solar = df[df["Technology Type"] == "Solar Photovoltaics"].copy()

    # Parse operational dates (format: "11/07/2011")
    def parse_operational_date(date_str):
        if pd.isna(date_str):
            return None
        try:
            return datetime.strptime(str(date_str).strip(), "%d/%m/%Y").year
        except ValueError:
            return None

    solar["operational_year"] = solar["Operational"].apply(parse_operational_date)

    # Create lookup by Old Ref ID (this is what OSM uses as repd:id)
    repd_lookup = {}
    for _, row in solar.iterrows():
        old_ref = row["Old Ref ID"]
        if pd.notna(old_ref):
            repd_lookup[str(int(old_ref))] = {
                "operational_year": row["operational_year"],
                "status": row["Development Status (short)"],
                "site_name": row["Site Name"],
            }

    print(f"Loaded {len(repd_lookup)} solar farms from REPD")
    operational_count = sum(1 for v in repd_lookup.values() if v["operational_year"] is not None)
    print(f"  {operational_count} have operational dates")

    return repd_lookup


def filter_features_by_repd(features_list: list, repd_lookup: dict, imagery_year: int) -> list:
    """
    Filter OSM features to only include those confirmed operational before imagery_year.

    Strategy:
    1. If feature has repd:id -> check REPD for operational date and status
    2. If feature has start_date in OSM -> parse and check
    3. If neither -> exclude (uncertain temporal alignment)
    4. Require operational_year < imagery_year (not just <=) to ensure fully established
    5. Check end_date for decommissioned farms
    """
    filtered = []
    stats = {
        "total": len(features_list),
        "repd_matched": 0,
        "repd_operational": 0,
        "osm_start_date": 0,
        "excluded_no_date": 0,
        "excluded_future": 0,
        "excluded_too_recent": 0,
        "excluded_decommissioned": 0,
        "excluded_not_operational": 0,
    }

    for feat in features_list:
        props = feat.get("properties", {})
        repd_id = props.get("repd:id")
        osm_start_date = props.get("start_date")
        osm_end_date = props.get("end_date")

        operational_year = None
        source = None
        repd_status = None

        # Strategy 1: Check REPD
        if repd_id:
            repd_id_str = str(repd_id).strip()
            if repd_id_str in repd_lookup:
                stats["repd_matched"] += 1
                repd_info = repd_lookup[repd_id_str]
                operational_year = repd_info["operational_year"]
                repd_status = repd_info.get("status")
                if operational_year:
                    source = "repd"

        # Strategy 2: Fall back to OSM start_date
        if operational_year is None and osm_start_date:
            parsed_year = parse_osm_start_date(osm_start_date)
            if parsed_year:
                operational_year = parsed_year
                source = "osm"

        # Decision: no date info
        if operational_year is None:
            stats["excluded_no_date"] += 1
            continue

        # Decision: built after imagery year
        if operational_year > imagery_year:
            stats["excluded_future"] += 1
            continue

        # Decision: built in the same year as imagery (may still be under construction)
        if operational_year == imagery_year:
            stats["excluded_too_recent"] += 1
            continue

        # Decision: check REPD status for non-operational entries
        if repd_status and repd_status in ("Under Construction", "Awaiting Construction", "Planning Permission Expired"):
            stats["excluded_not_operational"] += 1
            continue

        # Decision: check for decommissioned farms via OSM end_date
        if osm_end_date:
            end_year = parse_osm_start_date(osm_end_date)
            if end_year and end_year <= imagery_year:
                stats["excluded_decommissioned"] += 1
                continue

        # Keep this feature
        if source == "repd":
            stats["repd_operational"] += 1
        else:
            stats["osm_start_date"] += 1

        filtered.append(feat)

    print(f"\nREPD temporal filtering (imagery year: {imagery_year}):")
    print(f"  Total features: {stats['total']}")
    print(f"  REPD matched: {stats['repd_matched']}")
    print(f"  Kept (REPD operational): {stats['repd_operational']}")
    print(f"  Kept (OSM start_date): {stats['osm_start_date']}")
    print(f"  Excluded (no date info): {stats['excluded_no_date']}")
    print(f"  Excluded (future): {stats['excluded_future']}")
    print(f"  Excluded (too recent - same year): {stats['excluded_too_recent']}")
    print(f"  Excluded (decommissioned): {stats['excluded_decommissioned']}")
    print(f"  Excluded (not operational status): {stats['excluded_not_operational']}")
    print(f"  Final count: {len(filtered)}")

    return filtered


def download_osm_solar_farms(country_code: str = "GB", use_cache: bool = True) -> list:
    """Download solar farm polygons from OpenStreetMap with caching."""
    cache_path = Path(CACHE_FILE)

    # Check cache first
    if use_cache and cache_path.exists():
        print(f"Loading cached OSM data from {cache_path}")
        with open(cache_path) as f:
            osm_data = json.load(f)
        print(f"Loaded {len(osm_data['elements'])} OSM elements from cache")
    else:
        query = f"""
        [out:json][timeout:300];
        area["ISO3166-1"="{country_code}"]->.country;
        (
          way["power"="plant"]["plant:source"="solar"](area.country);
          relation["power"="plant"]["plant:source"="solar"](area.country);
        );
        out body;
        >;
        out skel qt;
        """

        print(f"Downloading solar farms from OpenStreetMap ({country_code})...")
        osm_data = None

        for endpoint in OVERPASS_ENDPOINTS:
            try:
                print(f"  Trying {endpoint}...")
                response = requests.post(endpoint, data={"data": query}, timeout=300)
                response.raise_for_status()
                osm_data = response.json()
                print(f"  Success!")
                break
            except requests.exceptions.RequestException as e:
                print(f"  Failed: {e}")
                continue

        if osm_data is None:
            raise RuntimeError("All Overpass endpoints failed. Try again later or use cached data.")

        print(f"Downloaded {len(osm_data['elements'])} OSM elements")

        # Cache the response
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(osm_data, f)
        print(f"Cached OSM data to {cache_path}")

    # Build node lookup
    nodes = {}
    for element in osm_data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])

    # Build way geometries
    ways = {}
    for element in osm_data["elements"]:
        if element["type"] == "way" and "nodes" in element:
            coords = [nodes[n] for n in element["nodes"] if n in nodes]
            if len(coords) >= 3:
                ways[element["id"]] = coords

    # Extract features
    features_list = []
    for element in osm_data["elements"]:
        if element["type"] == "way" and "tags" in element:
            if element["id"] in ways:
                coords = ways[element["id"]]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                geom = {"type": "Polygon", "coordinates": [coords]}
                features_list.append(geojson.Feature(
                    geometry=geom,
                    properties=element.get("tags", {}),
                    id=f"way/{element['id']}"
                ))
        elif element["type"] == "relation" and "tags" in element:
            if "members" in element:
                outer_rings = []
                for member in element["members"]:
                    if member["type"] == "way" and member.get("role") == "outer":
                        if member["ref"] in ways:
                            outer_rings.append(ways[member["ref"]])
                if outer_rings:
                    coords = outer_rings[0]
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    geom = {"type": "Polygon", "coordinates": [coords]}
                    features_list.append(geojson.Feature(
                        geometry=geom,
                        properties=element.get("tags", {}),
                        id=f"relation/{element['id']}"
                    ))

    print(f"Converted to {len(features_list)} polygon features")

    # Filter by area
    filtered = []
    for feat in features_list:
        try:
            geom = shape(feat["geometry"])
            lat = geom.centroid.y
            area_m2 = geom.area * (111000 ** 2) * abs(math.cos(math.radians(lat)))
            if area_m2 >= MIN_AREA_M2:
                filtered.append(feat)
        except Exception:
            continue

    print(f"After filtering (>= {MIN_AREA_M2} m²): {len(filtered)} features")
    return filtered


def longitude_split(features_list: list) -> dict:
    """Split features by longitude (left-to-right geographic split)."""
    # Sort by centroid longitude
    features_with_lon = []
    for feat in features_list:
        geom = shape(feat["geometry"])
        lon = geom.centroid.x
        features_with_lon.append((lon, feat))

    features_with_lon.sort(key=lambda x: x[0])
    sorted_features = [f for _, f in features_with_lon]

    n = len(sorted_features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        "train": sorted_features[:train_end],
        "val": sorted_features[train_end:val_end],
        "test": sorted_features[val_end:],
    }


def download_and_split(output_dir: Path, country_code: str = "GB", use_cache: bool = True, filter_by_repd: bool = True):
    """Download solar farms and create train/val/test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)

    features_list = download_osm_solar_farms(country_code, use_cache=use_cache)

    if not features_list:
        raise ValueError("No solar farms found!")

    # Filter by REPD operational dates
    if filter_by_repd:
        repd_lookup = load_repd_operational_dates(Path(REPD_FILE))
        features_list = filter_features_by_repd(features_list, repd_lookup, YEAR)

    if not features_list:
        raise ValueError("No solar farms remaining after REPD filtering!")

    # Save combined dataset
    combined_path = output_dir / "all_solar_farms.geojson"
    fc = geojson.FeatureCollection(features_list)
    with open(combined_path, "w") as f:
        geojson.dump(fc, f)
    print(f"Saved combined dataset to {combined_path}")

    # Split by longitude (left-to-right)
    splits = longitude_split(features_list)

    print("\nLongitude split (70/15/15):")
    for split, split_features in splits.items():
        print(f"  {split}: {len(split_features)} polygons")
        out_path = output_dir / f"{split}_solar_farms.geojson"
        fc = geojson.FeatureCollection(split_features)
        with open(out_path, "w") as f:
            geojson.dump(fc, f)

    return splits


def load_polygons(geojson_path: Path) -> list:
    """Load solar farm polygons from a GeoJSON file."""
    with open(geojson_path) as f:
        data = geojson.load(f)

    polygons = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        polygons.append({
            "geometry": geom,
            "bounds": geom.bounds,
        })

    print(f"Loaded {len(polygons)} polygons from {geojson_path}")
    return polygons


def transform_polygon(polygon, src_crs: str, dst_crs: str):
    """Transform a polygon between coordinate systems."""
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, polygon)


def compute_patch_transform(tile_transform, x_offset: int, y_offset: int):
    """Compute the transform for a patch given its offset within a tile."""
    return tile_transform * rasterio.Affine.translation(x_offset, y_offset)


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
        total_pixels = PATCH_SIZE * PATCH_SIZE
        for y in range(0, h - PATCH_SIZE + 1, PATCH_STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_STRIDE):
                emb_patch = embedding[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]
                mask_patch = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                if np.any(np.isnan(emb_patch)):
                    continue

                positive_pixels = mask_patch.sum()
                if positive_pixels < MIN_POSITIVE_PIXELS:
                    continue

                # Check coverage ratio to filter edge cases
                coverage = positive_pixels / total_pixels
                if coverage < MIN_COVERAGE_RATIO or coverage > MAX_COVERAGE_RATIO:
                    continue

                patch_transform = compute_patch_transform(tile_transform, x, y)
                patches.append((emb_patch.copy(), mask_patch.copy(), crs, patch_transform))

    return patches


def extract_negative_patches(gt: GeoTessera, num_patches: int, bounds: tuple) -> list:
    """Sample negative patches from tiles without solar farms."""
    patches = []

    all_tiles = gt.registry.load_blocks_for_region(bounds=bounds, year=YEAR)
    np.random.shuffle(all_tiles)

    patches_per_tile = max(10, num_patches // max(len(all_tiles), 1) + 1)

    for _, tile_lon, tile_lat in tqdm(all_tiles, desc="Sampling negative patches", leave=False):
        if len(patches) >= num_patches:
            break

        try:
            embedding, crs, tile_transform = gt.fetch_embedding(tile_lon, tile_lat, YEAR)
        except Exception:
            continue

        h, w, _ = embedding.shape
        empty_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

        if h < PATCH_SIZE or w < PATCH_SIZE:
            continue

        for _ in range(patches_per_tile):
            if len(patches) >= num_patches:
                break

            y = np.random.randint(0, h - PATCH_SIZE + 1)
            x = np.random.randint(0, w - PATCH_SIZE + 1)
            emb_patch = embedding[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]

            if not np.any(np.isnan(emb_patch)):
                patch_transform = compute_patch_transform(tile_transform, x, y)
                patches.append((emb_patch.copy(), empty_mask.copy(), crs, patch_transform))

    return patches


def prepare_patches(input_geojson: str, output_dir: str, stats_from: str = None):
    """Prepare patches from GeoJSON polygons."""
    input_path = Path(input_geojson)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gt = GeoTessera()
    polygons = load_polygons(input_path)

    all_bounds = [p["bounds"] for p in polygons]
    region_bounds = (
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    )
    print(f"Region bounds: {region_bounds}")

    # Extract positive patches
    all_patches = []
    print("Extracting patches around solar farms...")
    for poly_data in tqdm(polygons, desc="Processing polygons"):
        patches = extract_patches_for_polygon(gt, poly_data)
        all_patches.extend(patches)

    positive_count = len(all_patches)
    print(f"Extracted {positive_count} positive patches")

    # Extract negative patches (1:1 ratio)
    num_negative = positive_count
    print(f"Extracting {num_negative} negative patches (1:1 ratio)...")
    negative_patches = extract_negative_patches(gt, num_negative, region_bounds)
    all_patches.extend(negative_patches)

    negative_count = len(negative_patches)
    print(f"Extracted {negative_count} negative patches")

    # Save patches
    print(f"Saving {len(all_patches)} patches to {output_path}...")

    num_channels = all_patches[0][0].shape[-1] if all_patches else 128
    pixel_sum = np.zeros(num_channels, dtype=np.float64)
    pixel_sq_sum = np.zeros(num_channels, dtype=np.float64)
    total_pixels = 0

    for i, (emb, mask, crs, patch_transform) in enumerate(tqdm(all_patches, desc="Saving")):
        np.save(output_path / f"patch_{i:06d}_emb.npy", emb.astype(np.float32))
        np.save(output_path / f"patch_{i:06d}_mask.npy", mask.astype(np.uint8))

        with rasterio.open(
            output_path / f"patch_{i:06d}_mask.tif",
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=mask.dtype,
            crs=crs,
            transform=patch_transform,
        ) as dst:
            dst.write(mask, 1)

        # Export 10% sample of positive train tiles as GeoTIFFs for QGIS visualization
        if "train" in str(output_path) and mask.sum() > 0 and i % 10 == 0:
            train_tif_dir = Path("tmp/train_tifs")
            train_tif_dir.mkdir(parents=True, exist_ok=True)

            with rasterio.open(
                train_tif_dir / f"patch_{i:06d}_emb.tif",
                "w",
                driver="GTiff",
                height=emb.shape[0],
                width=emb.shape[1],
                count=emb.shape[2],
                dtype=np.float32,
                crs=crs,
                transform=patch_transform,
            ) as dst:
                for band_idx in range(emb.shape[2]):
                    dst.write(emb[:, :, band_idx], band_idx + 1)

            with rasterio.open(
                train_tif_dir / f"patch_{i:06d}_mask.tif",
                "w",
                driver="GTiff",
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=np.uint8,
                crs=crs,
                transform=patch_transform,
            ) as dst:
                dst.write(mask, 1)

        num_patch_pixels = emb.shape[0] * emb.shape[1]
        pixel_sum += emb.sum(axis=(0, 1))
        pixel_sq_sum += (emb ** 2).sum(axis=(0, 1))
        total_pixels += num_patch_pixels

    # Compute and save stats (only for training set)
    if stats_from is None:
        global_mean = pixel_sum / total_pixels
        global_var = (pixel_sq_sum / total_pixels) - (global_mean ** 2)
        global_std = np.sqrt(np.maximum(global_var, 0)) + 1e-8

        stats = {"mean": global_mean.tolist(), "std": global_std.tolist()}
        with open(output_path / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved normalization stats to {output_path / 'stats.json'}")

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
    import argparse

    parser = argparse.ArgumentParser(description="Prepare solar farm training data")
    parser.add_argument("--download", action="store_true", help="Download data from OSM (uses cache if available)")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh download, ignore cache")
    parser.add_argument("--no-repd-filter", action="store_true", help="Skip REPD temporal filtering")
    parser.add_argument("--country", default="GB", help="Country code for OSM download (default: GB)")
    args = parser.parse_args()

    data_dir = Path("data")

    if args.download:
        download_and_split(data_dir, args.country, use_cache=not args.no_cache, filter_by_repd=not args.no_repd_filter)

    # Prepare patches for each split
    prepare_patches("data/train_solar_farms.geojson", "patches/train")
    prepare_patches("data/val_solar_farms.geojson", "patches/val", stats_from="patches/train")
    prepare_patches("data/test_solar_farms.geojson", "patches/test", stats_from="patches/train")
