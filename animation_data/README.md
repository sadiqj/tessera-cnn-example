# Solar Farm UMAP Clustering Animation

This directory contains an interactive anime.js visualization showing how pixel embeddings cluster in UMAP space, colored by solar farm classification.

## Overview

The visualization shows:
- **Left panel**: Geographic/spatial view of sampled pixels in their original positions
- **Right panel**: UMAP-clustered view showing how pixels group by their feature embeddings
- **Color coding**:
  - 🟢 Green (with glow) = Solar farm pixels
  - 🔴 Red (with glow) = Non-solar pixels

## How to Run

### Step 1: Generate the Data

First, run the Python script to extract pixels, run CNN predictions, and apply UMAP clustering:

```bash
python create_umap_animation.py
```

This will:
1. Fetch a region with solar farms from GeoTessera
2. Run the trained CNN model to classify pixels
3. Sample 2000 pixels for visualization
4. Apply UMAP dimensionality reduction
5. Export data to `animation_data/umap_clustering.json`

**Note**: Make sure you have a trained model at `models/solar_unet.pth`. If not, run `python train.py` first.

### Step 2: View the Animation

Open `index.html` in a web browser:

```bash
# Option 1: Using Python's built-in server
cd animation_data
python -m http.server 8000
# Then open http://localhost:8000 in your browser

# Option 2: Open directly (may have CORS issues)
# Just double-click index.html or open it in your browser
```

### Step 3: Interact with the Visualization

- **Start Animation**: Click to see pixels morph from spatial positions to UMAP clusters
- **Reset**: Return pixels to their original spatial positions
- **Switch View**: Toggle between both views, spatial only, or UMAP only

## The Animation Sequence

1. **Initial State**: Points fade in at their geographic positions
2. **Phase 1**: Points shrink slightly in preparation
3. **Phase 2**: Points morph from spatial positions to UMAP clustering (staggered animation)
4. **Phase 3**: Points settle into their clustered positions with elastic easing
5. **Phase 4**: Solar farm and non-solar clusters pulse to highlight the separation

## Technical Details

### Data Format

The `umap_clustering.json` file contains:
```json
{
  "bbox": [lon_min, lat_min, lon_max, lat_max],
  "mosaic_shape": [height, width],
  "n_samples": 2000,
  "n_solar": 150,
  "n_non_solar": 1850,
  "threshold": 0.5,
  "points": [
    {
      "id": 0,
      "x": 100.5,          // Original pixel X coordinate
      "y": 50.2,           // Original pixel Y coordinate
      "umap_x": 0.234,     // UMAP X (normalized to 0-1)
      "umap_y": 0.876,     // UMAP Y (normalized to 0-1)
      "prediction": 0.92,  // CNN prediction probability
      "is_solar": true     // Binary classification
    },
    ...
  ]
}
```

### Animation Framework

- **anime.js v3.2.1**: Used for smooth, complex animations
- **HTML5 Canvas**: For high-performance rendering of 2000+ points
- **Staggered animations**: Points animate with delays from center outward
- **Elastic easing**: Creates natural, bouncy motion

## Customization

You can modify the Python script to:
- Change the region: Update `BBOX` in `create_umap_animation.py`
- Sample more/fewer pixels: Adjust `SAMPLE_SIZE`
- Tune UMAP parameters: Modify `n_neighbors` and `min_dist` in `apply_umap()`
- Change the year: Update `YEAR` variable

You can modify the HTML to:
- Adjust colors: Change the hex values in the CSS and JavaScript
- Change animation timing: Modify `duration` values in the `startAnimation()` function
- Customize point sizes: Adjust `baseRadius` in `drawPoint()`

## Requirements

### Python Dependencies
- geotessera
- torch
- numpy
- umap-learn (auto-installed if missing)

### Browser Requirements
- Modern browser with HTML5 Canvas support
- JavaScript enabled
- Internet connection (to load anime.js from CDN)

## Tips for Best Results

1. **Choose a good region**: Pick an area with a mix of solar farms and other land types
2. **Sufficient samples**: 2000 samples usually provides good clustering visualization
3. **Model quality**: Better trained models produce clearer clustering separation
4. **Screen size**: The visualization looks best on larger screens (1920x1080+)

## Troubleshooting

**Animation not loading?**
- Check browser console for errors
- Make sure `umap_clustering.json` exists in the same directory
- Try using a local HTTP server instead of opening file:// directly

**Clustering looks random?**
- Check that your CNN model is trained properly
- Try adjusting UMAP parameters (increase n_neighbors for smoother clusters)
- Ensure the region has actual solar farms

**Performance issues?**
- Reduce `SAMPLE_SIZE` in the Python script
- Close other browser tabs
- Try in Chrome/Edge (better Canvas performance)

## License

Part of the tessera-cnn-example solar farm detection project.
