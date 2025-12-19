#!/usr/bin/env python3
"""
Create animated WebP images showing cumulative solar farm development (2017-2024).
Solar farms accumulate - once detected in a year, they stay visible in subsequent frames.
"""

import json
import math
import os
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

# Configuration
UK_DATA_DIR = Path("uk")
OUTPUT_DIR = Path("animations")
YEARS = list(range(2017, 2025))

# EOX Sentinel-2 Cloudless tiles (WMTS)
# Available years: 2017-2024, using GoogleMapsCompatible (EPSG:3857) tile matrix
# URL format: https://tiles.maps.eox.at/wmts/1.0.0/{layer}/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg
EOX_TILE_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{year}_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg"
EOX_TILE_URL_LATEST = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg"

# Coordinate transformers
WGS84_TO_WEBMERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
WEBMERC_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def load_all_solar_farms() -> dict[int, gpd.GeoDataFrame]:
    """Load all solar farm polygons for each year."""
    year_data = {}

    for year in YEARS:
        year_dir = UK_DATA_DIR / str(year) / "tiles"
        if not year_dir.exists():
            print(f"Warning: No data for {year}")
            continue

        all_features = []
        geojson_files = list(year_dir.glob("*/polygons.geojson"))

        for geojson_path in tqdm(geojson_files, desc=f"Loading {year}"):
            try:
                with open(geojson_path) as f:
                    data = json.load(f)
                all_features.extend(data.get("features", []))
            except Exception as e:
                print(f"Error loading {geojson_path}: {e}")

        if all_features:
            gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
            gdf["year"] = year
            year_data[year] = gdf
            print(f"  {year}: {len(gdf)} farms")

    return year_data


def find_best_regions(year_data: dict[int, gpd.GeoDataFrame],
                      cell_size: float = 0.5,
                      top_n: int = 3) -> list[dict]:
    """
    Find regions with highest solar farm density and growth.

    Args:
        year_data: Dict mapping year to GeoDataFrame of farms
        cell_size: Size of grid cells in degrees
        top_n: Number of top regions to return

    Returns:
        List of dicts with 'name', 'bbox', 'total_farms', 'growth'
    """
    # Combine all farms to get total extent
    all_farms = []
    for gdf in year_data.values():
        all_farms.append(gdf)
    combined = gpd.GeoDataFrame(pd.concat(all_farms, ignore_index=True), crs="EPSG:4326")

    # Get bounds
    minx, miny, maxx, maxy = combined.total_bounds

    # Create grid cells
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell_bbox = box(x, y, x + cell_size, y + cell_size)
            cells.append({
                'bbox': (x, y, x + cell_size, y + cell_size),
                'geometry': cell_bbox
            })
            y += cell_size
        x += cell_size

    # Count farms per cell per year
    cell_stats = []
    for cell in tqdm(cells, desc="Analyzing grid cells"):
        cell_geom = cell['geometry']
        yearly_counts = {}

        for year, gdf in year_data.items():
            # Count farms whose centroid is in this cell
            centroids = gdf.geometry.centroid
            count = centroids.within(cell_geom).sum()
            yearly_counts[year] = count

        total = sum(yearly_counts.values())
        if total == 0:
            continue

        # Calculate growth (difference between last and first year with data)
        years_with_data = [y for y, c in yearly_counts.items() if c > 0]
        if len(years_with_data) >= 2:
            first_year = min(years_with_data)
            last_year = max(years_with_data)
            growth = yearly_counts[last_year] - yearly_counts[first_year]
        else:
            growth = 0

        # 2024 count for final density
        final_count = yearly_counts.get(2024, 0)

        cell_stats.append({
            'bbox': cell['bbox'],
            'total_farms': total,
            'final_count': final_count,
            'growth': growth,
            'yearly_counts': yearly_counts,
            'center': ((cell['bbox'][0] + cell['bbox'][2]) / 2,
                      (cell['bbox'][1] + cell['bbox'][3]) / 2)
        })

    # Score cells: prioritize density and growth
    for cell in cell_stats:
        # Score = final_count + growth bonus
        cell['score'] = cell['final_count'] * 2 + cell['growth']

    # Sort by score and get top regions
    cell_stats.sort(key=lambda x: x['score'], reverse=True)

    # Get unique top regions (avoid overlapping regions)
    selected = []
    for cell in cell_stats:
        if len(selected) >= top_n:
            break

        # Check if this cell is too close to already selected ones
        center = cell['center']
        too_close = False
        for sel in selected:
            sel_center = sel['center']
            dist = math.sqrt((center[0] - sel_center[0])**2 + (center[1] - sel_center[1])**2)
            if dist < cell_size * 1.5:  # Minimum separation
                too_close = True
                break

        if not too_close:
            selected.append(cell)

    # Name the regions based on approximate location
    region_names = get_region_names([c['center'] for c in selected])

    results = []
    for i, cell in enumerate(selected):
        results.append({
            'name': region_names[i],
            'bbox': cell['bbox'],
            'total_farms': cell['total_farms'],
            'final_count': cell['final_count'],
            'growth': cell['growth'],
            'yearly_counts': cell['yearly_counts']
        })

    return results


def get_region_names(centers: list[tuple]) -> list[str]:
    """Get approximate region names based on coordinates."""
    # Rough UK region mapping
    regions = []
    for lon, lat in centers:
        if lat > 54:
            region = "Northern England"
        elif lat > 53:
            if lon < -1.5:
                region = "Lancashire"
            else:
                region = "Yorkshire"
        elif lat > 52:
            if lon < -1:
                region = "West Midlands"
            else:
                region = "East Midlands"
        elif lat > 51.5:
            if lon < -1:
                region = "Oxfordshire"
            elif lon > 0.5:
                region = "East Anglia"
            else:
                region = "Cambridgeshire"
        elif lat > 51:
            if lon < -2:
                region = "Somerset"
            elif lon < 0:
                region = "Hampshire"
            else:
                region = "Kent"
        else:
            if lon < -3:
                region = "Cornwall"
            elif lon < -2:
                region = "Devon"
            else:
                region = "South Coast"

        # Make unique if needed
        base = region
        counter = 2
        while region in regions:
            region = f"{base} {counter}"
            counter += 1
        regions.append(region)

    return regions


def deg2tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tile2deg(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert tile coordinates to lat/lon (northwest corner)."""
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def fetch_satellite_image(bbox: tuple, zoom: int = 15,
                          target_size: tuple = (1200, 900),
                          year: int = None) -> tuple[Image.Image, tuple]:
    """
    Fetch and stitch satellite tiles for the given bounding box.
    Uses EOX Sentinel-2 Cloudless imagery.

    Args:
        bbox: (minx, miny, maxx, maxy) in WGS84
        zoom: Tile zoom level
        target_size: Target image size (width, height)
        year: Optional year for year-specific imagery (2017-2024)

    Returns:
        Tuple of (PIL Image, actual_bbox)
    """
    minx, miny, maxx, maxy = bbox

    # Get tile range
    x_min, y_max = deg2tile(miny, minx, zoom)
    x_max, y_min = deg2tile(maxy, maxx, zoom)

    # Expand slightly to ensure coverage
    x_min -= 1
    y_min -= 1
    x_max += 1
    y_max += 1

    # Calculate actual bbox from tiles
    nw_lat, nw_lon = tile2deg(x_min, y_min, zoom)
    se_lat, se_lon = tile2deg(x_max + 1, y_max + 1, zoom)
    actual_bbox = (nw_lon, se_lat, se_lon, nw_lat)

    # Fetch tiles
    tile_size = 256
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size

    combined = Image.new('RGB', (width, height))

    session = requests.Session()

    # Select tile URL based on year
    if year and 2017 <= year <= 2024:
        tile_url_template = EOX_TILE_URL.format(year=year, z="{z}", y="{y}", x="{x}")
    else:
        tile_url_template = EOX_TILE_URL_LATEST

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # EOX WMTS uses z/y/x (row/col) format
            url = tile_url_template.format(z=zoom, y=y, x=x)

            try:
                resp = session.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; SolarFarmViz/1.0)'
                })
                if resp.status_code == 200:
                    tile_img = Image.open(BytesIO(resp.content))
                    px = (x - x_min) * tile_size
                    py = (y - y_min) * tile_size
                    combined.paste(tile_img, (px, py))
                else:
                    print(f"  Warning: Tile {x},{y} returned status {resp.status_code}")
            except Exception as e:
                print(f"  Warning: Failed to fetch tile {x},{y}: {e}")

    # Resize to target size while maintaining aspect ratio
    combined.thumbnail(target_size, Image.Resampling.LANCZOS)

    return combined, actual_bbox


def render_frame(satellite_img: Image.Image,
                 actual_bbox: tuple,
                 farms_gdf: gpd.GeoDataFrame,
                 year: int,
                 new_farms_gdf: gpd.GeoDataFrame = None,
                 target_size: tuple = (1200, 900)) -> Image.Image:
    """
    Render a single frame with satellite backdrop and farm polygons.

    Args:
        satellite_img: Background satellite image
        actual_bbox: Bounding box of the satellite image
        farms_gdf: GeoDataFrame of all farms to show (cumulative)
        year: Year to display
        new_farms_gdf: GeoDataFrame of farms new this year (for highlighting)
        target_size: Output image size (width, height)

    Returns:
        PIL Image of the rendered frame
    """
    # Use fixed DPI and figure size for consistent output
    dpi = 100
    fig_width = target_size[0] / dpi
    fig_height = target_size[1] / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)

    # Show satellite backdrop (50% transparent)
    minx, miny, maxx, maxy = actual_bbox
    ax.imshow(satellite_img, extent=[minx, maxx, miny, maxy], aspect='auto', alpha=0.5)

    # Clip farms to bbox
    bbox_geom = box(minx, miny, maxx, maxy)

    if len(farms_gdf) > 0:
        farms_clipped = farms_gdf[farms_gdf.geometry.intersects(bbox_geom)]

        # Draw existing farms (semi-transparent yellow)
        if len(farms_clipped) > 0:
            farms_clipped.plot(ax=ax, color='#FFD700', alpha=0.6, edgecolor='#FF8C00', linewidth=0.5)

    # Highlight new farms if provided
    if new_farms_gdf is not None and len(new_farms_gdf) > 0:
        new_clipped = new_farms_gdf[new_farms_gdf.geometry.intersects(bbox_geom)]
        if len(new_clipped) > 0:
            new_clipped.plot(ax=ax, color='#00FF00', alpha=0.8, edgecolor='#00AA00', linewidth=1)

    # Set bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    # Add year label with shadow effect
    text_x = minx + (maxx - minx) * 0.05
    text_y = maxy - (maxy - miny) * 0.08

    # Shadow
    ax.text(text_x + 0.002, text_y - 0.002, str(year),
            fontsize=48, fontweight='bold', color='black', alpha=0.7)
    # Main text
    ax.text(text_x, text_y, str(year),
            fontsize=48, fontweight='bold', color='white')

    # Add farm count
    count_text = f"{len(farms_gdf)} farms" if len(farms_gdf) > 0 else "0 farms"
    ax.text(text_x + 0.002, text_y - (maxy - miny) * 0.06 - 0.002, count_text,
            fontsize=20, fontweight='bold', color='black', alpha=0.7)
    ax.text(text_x, text_y - (maxy - miny) * 0.06, count_text,
            fontsize=20, fontweight='bold', color='white')

    # Remove all margins
    ax.set_position([0, 0, 1, 1])

    # Convert to PIL Image with fixed size
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='black', edgecolor='none')
    buf.seek(0)
    frame = Image.open(buf).convert('RGB')
    plt.close(fig)

    # Ensure exact target size
    if frame.size != target_size:
        frame = frame.resize(target_size, Image.Resampling.LANCZOS)

    return frame


def create_animation(region_name: str,
                     bbox: tuple,
                     year_data: dict[int, gpd.GeoDataFrame],
                     output_path: Path,
                     zoom: int = 14,
                     frame_duration: int = 1000,
                     use_yearly_imagery: bool = True) -> None:
    """
    Create animated WebP for a region showing cumulative solar farm development.

    Args:
        region_name: Name of the region for display
        bbox: Bounding box (minx, miny, maxx, maxy)
        year_data: Dict mapping year to GeoDataFrame
        output_path: Path for output WebP file
        zoom: Tile zoom level
        frame_duration: Duration of each frame in milliseconds
        use_yearly_imagery: If True, fetch year-specific Sentinel-2 imagery for each frame
    """
    print(f"\nCreating animation for {region_name}...")

    # Build cumulative farms for each year
    frames = []
    cumulative_farms = []
    actual_bbox = None
    satellite_cache = {}

    for year in YEARS:
        print(f"  Rendering {year}...")

        # Fetch satellite imagery (year-specific or cached)
        if use_yearly_imagery:
            if year not in satellite_cache:
                print(f"    Fetching {year} satellite tiles...")
                satellite_img, year_bbox = fetch_satellite_image(bbox, zoom=zoom, year=year)
                satellite_cache[year] = satellite_img
                # Use first year's bbox for consistency across all frames
                if actual_bbox is None:
                    actual_bbox = year_bbox
            else:
                satellite_img = satellite_cache[year]
        else:
            # Use single cached image
            if not satellite_cache:
                print("  Fetching satellite tiles...")
                satellite_img, actual_bbox = fetch_satellite_image(bbox, zoom=zoom)
                satellite_cache['static'] = satellite_img
            else:
                satellite_img = satellite_cache['static']

        # Get farms for this year
        if year in year_data:
            year_farms = year_data[year]
            # Filter to bbox
            bbox_geom = box(*actual_bbox)
            year_farms_in_bbox = year_farms[year_farms.geometry.intersects(bbox_geom)]
        else:
            year_farms_in_bbox = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Add new farms to cumulative (avoid duplicates by spatial similarity)
        new_this_year = []
        for _, farm in year_farms_in_bbox.iterrows():
            # Check if similar farm already exists
            is_new = True
            for existing in cumulative_farms:
                if farm.geometry.intersection(existing.geometry).area > farm.geometry.area * 0.5:
                    is_new = False
                    break
            if is_new:
                new_this_year.append(farm)
                cumulative_farms.append(farm)

        # Create GeoDataFrames
        if cumulative_farms:
            cumulative_gdf = gpd.GeoDataFrame(cumulative_farms, crs="EPSG:4326")
        else:
            cumulative_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if new_this_year:
            new_gdf = gpd.GeoDataFrame(new_this_year, crs="EPSG:4326")
        else:
            new_gdf = None

        # Render frame
        frame = render_frame(satellite_img, actual_bbox, cumulative_gdf, year, new_gdf)
        frames.append(frame)

    # Save as animated WebP
    print(f"  Saving animation to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,  # Infinite loop
        quality=85
    )
    print(f"  Done! Created {output_path}")


def main():
    import pandas as pd  # Import here to avoid issues if not available

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Solar Farm Animation Generator")
    print("=" * 60)

    # Load all data
    print("\nStep 1: Loading solar farm data...")
    year_data = load_all_solar_farms()

    if not year_data:
        print("Error: No data found!")
        return

    # Find best regions
    print("\nStep 2: Finding best regions for visualization...")
    # Import pandas here since we need it for concatenation
    import pandas as pd
    globals()['pd'] = pd

    best_regions = find_best_regions(year_data, cell_size=0.3, top_n=10)

    print("\nTop regions found:")
    for i, region in enumerate(best_regions, 1):
        print(f"  {i}. {region['name']}")
        print(f"     - Total farms: {region['total_farms']}")
        print(f"     - Final count (2024): {region['final_count']}")
        print(f"     - Growth: +{region['growth']}")
        print(f"     - Bbox: {region['bbox']}")

    # Create animations for each region
    print("\nStep 3: Creating animations...")
    for region in best_regions:
        output_path = OUTPUT_DIR / f"{region['name'].lower().replace(' ', '_')}_solar_farms.webp"
        create_animation(
            region_name=region['name'],
            bbox=region['bbox'],
            year_data=year_data,
            output_path=output_path,
            zoom=13,  # Zoomed out 1.5x from 14
            frame_duration=1200,  # 1.2 seconds per frame
            use_yearly_imagery=False  # Use static 2024 basemap
        )

    print("\n" + "=" * 60)
    print("All animations created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
