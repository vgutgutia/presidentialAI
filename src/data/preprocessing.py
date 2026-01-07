"""
Preprocessing utilities for Sentinel-2 imagery.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from pathlib import Path


def normalize_bands(
    image: np.ndarray,
    mean: List[float],
    std: List[float],
) -> np.ndarray:
    """
    Normalize image bands using mean and standard deviation.
    
    Args:
        image: Input image of shape (C, H, W)
        mean: Per-band means
        std: Per-band standard deviations
        
    Returns:
        Normalized image
    """
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    
    return (image - mean) / (std + 1e-8)


def denormalize_bands(
    image: np.ndarray,
    mean: List[float],
    std: List[float],
) -> np.ndarray:
    """
    Denormalize image bands.
    
    Args:
        image: Normalized image of shape (C, H, W)
        mean: Per-band means
        std: Per-band standard deviations
        
    Returns:
        Denormalized image
    """
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    
    return image * std + mean


def resample_band(
    band: np.ndarray,
    source_resolution: float,
    target_resolution: float,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Resample a band to target resolution.
    
    Args:
        band: Input band array (H, W)
        source_resolution: Source resolution in meters
        target_resolution: Target resolution in meters
        method: Resampling method ('nearest', 'bilinear', 'cubic')
        
    Returns:
        Resampled band
    """
    if source_resolution == target_resolution:
        return band
    
    scale = source_resolution / target_resolution
    new_height = int(band.shape[0] * scale)
    new_width = int(band.shape[1] * scale)
    
    # Use scipy for simple resampling
    from scipy.ndimage import zoom
    
    order = {"nearest": 0, "bilinear": 1, "cubic": 3}.get(method, 1)
    
    return zoom(band, scale, order=order)


def preprocess_scene(
    scene_path: str,
    output_path: str,
    bands: List[str] = None,
    target_resolution: float = 10.0,
    clip_values: Tuple[float, float] = (0, 10000),
    scale_factor: float = 10000.0,
) -> str:
    """
    Preprocess a Sentinel-2 scene for inference.
    
    Args:
        scene_path: Path to input scene (GeoTIFF or SAFE format)
        output_path: Path to save preprocessed scene
        bands: List of bands to extract
        target_resolution: Target resolution in meters
        clip_values: Min/max values for clipping
        scale_factor: Scale factor for reflectance values
        
    Returns:
        Path to preprocessed scene
    """
    bands = bands or ["B2", "B3", "B4", "B8", "B11", "B12"]
    
    scene_path = Path(scene_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Band resolutions in Sentinel-2
    BAND_RESOLUTIONS = {
        "B1": 60, "B2": 10, "B3": 10, "B4": 10, "B5": 20,
        "B6": 20, "B7": 20, "B8": 10, "B8A": 20, "B9": 60,
        "B10": 60, "B11": 20, "B12": 20,
    }
    
    if scene_path.suffix == ".tif":
        # Single GeoTIFF file - assume bands are already stacked
        with rasterio.open(scene_path) as src:
            # Read all bands
            data = src.read().astype(np.float32)
            profile = src.profile.copy()
            
            # Clip and scale
            data = np.clip(data, clip_values[0], clip_values[1])
            data = data / scale_factor
            
            # Update profile
            profile.update(
                dtype=rasterio.float32,
                count=data.shape[0],
            )
            
            # Save
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)
    else:
        raise ValueError(f"Unsupported file format: {scene_path.suffix}")
    
    return str(output_path)


def load_sentinel2_scene(
    path: str,
    bands: List[str] = None,
    normalize: bool = True,
    normalization: Optional[Dict[str, List[float]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a Sentinel-2 scene.
    
    Args:
        path: Path to GeoTIFF file
        bands: List of bands to load (assumes bands in order)
        normalize: Whether to normalize the data
        normalization: Normalization statistics
        
    Returns:
        Tuple of (image array, metadata dict)
    """
    bands = bands or ["B2", "B3", "B4", "B8", "B11", "B12"]
    
    with rasterio.open(path) as src:
        # Read data
        n_bands = min(src.count, len(bands))
        data = src.read(list(range(1, n_bands + 1))).astype(np.float32)
        
        # Get metadata
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "height": src.height,
            "width": src.width,
            "resolution": src.res,
        }
    
    # Normalize if requested
    if normalize and normalization:
        data = normalize_bands(
            data,
            mean=normalization["mean"][:n_bands],
            std=normalization["std"][:n_bands]
        )
    
    return data, metadata


def compute_band_statistics(
    data_dir: str,
    bands: List[str] = None,
    sample_size: int = 1000,
) -> Dict[str, List[float]]:
    """
    Compute mean and std statistics across dataset.
    
    Args:
        data_dir: Path to dataset directory
        bands: List of bands
        sample_size: Number of patches to sample
        
    Returns:
        Dict with 'mean' and 'std' lists
    """
    bands = bands or ["B2", "B3", "B4", "B8", "B11", "B12"]
    n_bands = len(bands)
    
    data_dir = Path(data_dir)
    patches_dir = data_dir / "patches"
    
    if not patches_dir.exists():
        patches_dir = data_dir / "scenes"
    
    # Collect statistics
    all_means = []
    all_stds = []
    
    files = list(patches_dir.glob("*.tif"))[:sample_size]
    
    for filepath in files:
        with rasterio.open(filepath) as src:
            data = src.read(list(range(1, n_bands + 1))).astype(np.float32)
            
            # Per-band statistics
            for i in range(n_bands):
                band_data = data[i]
                valid_mask = band_data > 0  # Ignore nodata
                
                if valid_mask.sum() > 0:
                    all_means.append(band_data[valid_mask].mean())
                    all_stds.append(band_data[valid_mask].std())
    
    # Aggregate
    means = [float(np.mean(all_means[i::n_bands])) for i in range(n_bands)]
    stds = [float(np.mean(all_stds[i::n_bands])) for i in range(n_bands)]
    
    return {"mean": means, "std": stds}


def create_tiles(
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split an image into overlapping tiles.
    
    Args:
        image: Input image of shape (C, H, W)
        tile_size: Size of tiles
        overlap: Overlap between tiles
        
    Returns:
        List of (tile, (row, col)) tuples
    """
    _, height, width = image.shape
    stride = tile_size - overlap
    
    tiles = []
    
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            # Extract tile
            row_end = min(row + tile_size, height)
            col_end = min(col + tile_size, width)
            
            tile = image[:, row:row_end, col:col_end]
            
            # Pad if necessary
            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                padded = np.zeros((image.shape[0], tile_size, tile_size), dtype=image.dtype)
                padded[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = padded
            
            tiles.append((tile, (row, col)))
    
    return tiles


def stitch_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    output_shape: Tuple[int, int],
    tile_size: int = 256,
    overlap: int = 64,
) -> np.ndarray:
    """
    Stitch tiles back into a full image.
    
    Uses weighted averaging in overlap regions.
    
    Args:
        tiles: List of (tile, (row, col)) tuples
        output_shape: Shape of output image (H, W)
        tile_size: Size of tiles
        overlap: Overlap between tiles
        
    Returns:
        Stitched image
    """
    height, width = output_shape
    
    # Determine number of channels from first tile
    n_channels = tiles[0][0].shape[0] if tiles[0][0].ndim == 3 else 1
    
    # Initialize output and weight arrays
    if n_channels > 1:
        output = np.zeros((n_channels, height, width), dtype=np.float32)
    else:
        output = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    
    # Create weight mask for blending
    weight_mask = _create_weight_mask(tile_size, overlap)
    
    for tile, (row, col) in tiles:
        # Calculate actual tile size (may be smaller at edges)
        h = min(tile_size, height - row)
        w = min(tile_size, width - col)
        
        # Get weight for this region
        w_region = weight_mask[:h, :w]
        
        # Add weighted tile to output
        if n_channels > 1:
            for c in range(n_channels):
                output[c, row:row+h, col:col+w] += tile[c, :h, :w] * w_region
        else:
            output[row:row+h, col:col+w] += tile[:h, :w] * w_region
        
        weights[row:row+h, col:col+w] += w_region
    
    # Normalize by weights
    weights = np.maximum(weights, 1e-8)
    
    if n_channels > 1:
        for c in range(n_channels):
            output[c] /= weights
    else:
        output /= weights
    
    return output


def _create_weight_mask(tile_size: int, overlap: int) -> np.ndarray:
    """Create a weight mask for tile blending."""
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Create linear ramps at edges
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        
        # Apply ramps to edges
        mask[:overlap, :] *= ramp.reshape(-1, 1)
        mask[-overlap:, :] *= ramp[::-1].reshape(-1, 1)
        mask[:, :overlap] *= ramp.reshape(1, -1)
        mask[:, -overlap:] *= ramp[::-1].reshape(1, -1)
    
    return mask
