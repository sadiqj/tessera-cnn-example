#!/usr/bin/env python3
"""
Fetch solar farm polygons from OpenStreetMap and split them into train/test sets.
"""

import json
import os
from pathlib import Path
import requests
from geojson import Feature, FeatureCollection, Polygon, MultiPolygon


# Bounding box coordinates
NORTH = 52.7213
SOUTH = 51.4639
WEST = -1.6656
EAST = 0.3438

# Calculate midpoint longitude for train/test split
MID_LON = (WEST + EAST) / 2

# Overpass API endpoint - using alternative instance
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"


def build_overpass_query():
    """Build Overpass QL query for solar farms in the bounding box."""
    bbox = f"{SOUTH},{WEST},{NORTH},{EAST}"

    query = f"""
    [out:json][timeout:180];
    (
      // Query for solar power plants with photovoltaic method
      way["power"="plant"]["plant:source"="solar"]["plant:method"="photovoltaic"]({bbox});
      relation["power"="plant"]["plant:source"="solar"]["plant:method"="photovoltaic"]({bbox});

      // Also include plants that have plant:source=solar but may not specify method
      way["power"="plant"]["plant:source"="solar"]({bbox});
      relation["power"="plant"]["plant:source"="solar"]({bbox});
    );
    out geom;
    """
    return query


def fetch_solar_farms():
    """Fetch solar farm data from OpenStreetMap via Overpass API."""
    query = build_overpass_query()

    print("Querying OpenStreetMap for solar farms...")
    print(f"Bounding box: ({SOUTH}, {WEST}) to ({NORTH}, {EAST})")

    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=200
    )
    response.raise_for_status()

    data = response.json()
    print(f"Found {len(data.get('elements', []))} elements")

    return data


def convert_to_geojson_polygon(element):
    """Convert an OSM element to a GeoJSON polygon."""
    geometry = element.get("geometry", [])

    if not geometry:
        return None

    # Extract coordinates
    coords = [(point["lon"], point["lat"]) for point in geometry]

    # Close the polygon if not already closed
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(coords) < 4:  # A valid polygon needs at least 4 points (including closing point)
        return None

    return Polygon([coords])


def convert_to_geojson_multipolygon(element):
    """Convert an OSM relation to a GeoJSON multipolygon."""
    members = element.get("members", [])

    if not members:
        return None

    # For simplicity, we'll try to extract geometry from the members
    # This is a simplified version - proper OSM relation handling is complex
    polygons = []

    for member in members:
        if member.get("role") == "outer" and "geometry" in member:
            coords = [(point["lon"], point["lat"]) for point in member["geometry"]]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            if len(coords) >= 4:
                polygons.append([coords])

    if not polygons:
        return None

    if len(polygons) == 1:
        return Polygon(polygons[0])
    else:
        return MultiPolygon([poly for poly in polygons])


def calculate_centroid(geometry):
    """Calculate the centroid longitude of a geometry."""
    if geometry["type"] == "Polygon":
        coords = geometry["coordinates"][0]
        avg_lon = sum(c[0] for c in coords) / len(coords)
        return avg_lon
    elif geometry["type"] == "MultiPolygon":
        all_coords = []
        for polygon in geometry["coordinates"]:
            all_coords.extend(polygon[0])
        avg_lon = sum(c[0] for c in all_coords) / len(all_coords)
        return avg_lon
    return None


def process_solar_farms(data):
    """Process OSM data and split into train/test sets."""
    train_features = []
    test_features = []

    for element in data.get("elements", []):
        # Convert to GeoJSON geometry
        geometry = None

        if element["type"] == "way":
            geometry = convert_to_geojson_polygon(element)
        elif element["type"] == "relation":
            geometry = convert_to_geojson_multipolygon(element)

        if geometry is None:
            continue

        # Create feature with properties
        properties = {
            "osm_id": element.get("id"),
            "osm_type": element.get("type"),
            "tags": element.get("tags", {}),
        }

        feature = Feature(geometry=geometry, properties=properties)

        # Calculate centroid longitude for splitting
        centroid_lon = calculate_centroid(geometry)

        if centroid_lon is not None:
            # Split based on longitude
            if centroid_lon < MID_LON:
                train_features.append(feature)
            else:
                test_features.append(feature)

    print(f"Split results: {len(train_features)} train, {len(test_features)} test")

    return train_features, test_features


def save_geojson(features, filepath):
    """Save features to a GeoJSON file."""
    feature_collection = FeatureCollection(features)

    with open(filepath, "w") as f:
        json.dump(feature_collection, f, indent=2)

    print(f"Saved {len(features)} features to {filepath}")


def main():
    """Main function to fetch and process solar farm data."""
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Fetch data from OpenStreetMap
    data = fetch_solar_farms()

    # Process and split into train/test
    train_features, test_features = process_solar_farms(data)

    # Save to GeoJSON files
    save_geojson(train_features, data_dir / "train_solar_farms.geojson")
    save_geojson(test_features, data_dir / "test_solar_farms.geojson")

    print("\nDone! Solar farm data fetched and split successfully.")


if __name__ == "__main__":
    main()
