"""
Data download utilities for MARIDA dataset and Sentinel-2 imagery.
"""

import os
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import hashlib

import requests
from tqdm import tqdm


# MARIDA dataset download info
MARIDA_URLS = {
    "patches": "https://zenodo.org/record/5151941/files/patches.zip",
    "masks": "https://zenodo.org/record/5151941/files/masks.zip",
    "splits": "https://zenodo.org/record/5151941/files/splits.zip",
}

MARIDA_GITHUB = "https://github.com/marine-debris/marine-debris.github.io"


def download_file(
    url: str,
    output_path: str,
    chunk_size: int = 8192,
    show_progress: bool = True,
) -> str:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save file
        chunk_size: Download chunk size
        show_progress: Whether to show progress bar
        
    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get file size
    response = requests.head(url, allow_redirects=True)
    total_size = int(response.headers.get("content-length", 0))
    
    # Download
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        if show_progress and total_size:
            pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            )
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                if show_progress and total_size:
                    pbar.update(len(chunk))
        
        if show_progress and total_size:
            pbar.close()
    
    return str(output_path)


def extract_archive(
    archive_path: str,
    output_dir: str,
    remove_archive: bool = False,
) -> str:
    """
    Extract a zip or tar archive.
    
    Args:
        archive_path: Path to archive file
        output_dir: Directory to extract to
        remove_archive: Whether to delete archive after extraction
        
    Returns:
        Path to extracted directory
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {archive_path.name}...")
    
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    if remove_archive:
        archive_path.unlink()
    
    return str(output_dir)


def download_marida(
    output_dir: str = "data/marida",
    force_download: bool = False,
) -> str:
    """
    Download the MARIDA dataset.
    
    The MARIDA dataset contains Sentinel-2 imagery patches with pixel-level
    annotations for marine debris detection.
    
    Args:
        output_dir: Directory to save dataset
        force_download: Re-download even if exists
        
    Returns:
        Path to dataset directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if (output_dir / "patches").exists() and not force_download:
        print(f"MARIDA dataset already exists at {output_dir}")
        return str(output_dir)
    
    print("=" * 60)
    print("Downloading MARIDA Dataset")
    print("=" * 60)
    print(f"This dataset is hosted on Zenodo and GitHub.")
    print(f"Please cite: Kikaki et al. (2022) - MARIDA dataset")
    print("=" * 60)
    
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download each component
        for name, url in MARIDA_URLS.items():
            print(f"\nDownloading {name}...")
            
            archive_path = temp_dir / f"{name}.zip"
            
            try:
                download_file(url, str(archive_path))
                extract_archive(str(archive_path), str(output_dir), remove_archive=True)
            except Exception as e:
                print(f"Failed to download from Zenodo: {e}")
                print("Please download manually from:")
                print(f"  {url}")
                print(f"  or from GitHub: {MARIDA_GITHUB}")
        
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print(f"\nMARIDA dataset downloaded to: {output_dir}")
        
        # Verify structure
        expected_dirs = ["patches", "masks"]
        for d in expected_dirs:
            if not (output_dir / d).exists():
                print(f"Warning: Expected directory '{d}' not found.")
                print("Dataset structure may differ. Check manually.")
        
    except Exception as e:
        print(f"\nError downloading MARIDA: {e}")
        print("\nAlternative download instructions:")
        print("1. Visit: https://zenodo.org/record/5151941")
        print("2. Download patches.zip and masks.zip")
        print(f"3. Extract to: {output_dir}")
        raise
    
    return str(output_dir)


def download_sentinel2_scene(
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    output_dir: str = "data/raw",
    max_cloud_cover: float = 20.0,
    bands: List[str] = None,
) -> Optional[str]:
    """
    Download a Sentinel-2 scene from Microsoft Planetary Computer.
    
    Args:
        bbox: Bounding box (west, south, east, north) in WGS84
        date_range: Date range (start_date, end_date) as 'YYYY-MM-DD'
        output_dir: Directory to save scene
        max_cloud_cover: Maximum cloud cover percentage
        bands: List of bands to download
        
    Returns:
        Path to downloaded scene, or None if no scene found
    """
    bands = bands or ["B02", "B03", "B04", "B08", "B11", "B12"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Searching for Sentinel-2 scenes...")
    print(f"  Bbox: {bbox}")
    print(f"  Date range: {date_range}")
    print(f"  Max cloud cover: {max_cloud_cover}%")
    
    try:
        from pystac_client import Client
        import planetary_computer as pc
        import stackstac
        
        # Connect to Planetary Computer
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        
        # Search for scenes
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        
        items = list(search.items())
        
        if not items:
            print("No scenes found matching criteria.")
            return None
        
        print(f"Found {len(items)} scenes. Using most recent...")
        
        # Sort by date and get most recent
        items.sort(key=lambda x: x.properties["datetime"], reverse=True)
        item = items[0]
        
        print(f"Selected scene: {item.id}")
        print(f"  Date: {item.properties['datetime']}")
        print(f"  Cloud cover: {item.properties['eo:cloud_cover']:.1f}%")
        
        # Load as DataArray
        data = stackstac.stack(
            [item],
            assets=bands,
            bounds_latlon=bbox,
            resolution=10,
        )
        
        # Compute and save
        print("Downloading and processing...")
        result = data.compute()
        
        # Save as GeoTIFF
        output_path = output_dir / f"{item.id}.tif"
        
        # Convert to numpy and save
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
        
        arr = result.values[0]  # Remove time dimension
        
        transform = from_bounds(*bbox, arr.shape[2], arr.shape[1])
        
        profile = {
            "driver": "GTiff",
            "dtype": arr.dtype,
            "width": arr.shape[2],
            "height": arr.shape[1],
            "count": arr.shape[0],
            "crs": CRS.from_epsg(4326),
            "transform": transform,
            "compress": "lzw",
        }
        
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(arr)
        
        print(f"Saved to: {output_path}")
        return str(output_path)
        
    except ImportError:
        print("Required packages not installed for Planetary Computer access.")
        print("Install with: pip install pystac-client planetary-computer stackstac")
        return None
    except Exception as e:
        print(f"Error downloading scene: {e}")
        return None


def create_sample_data(output_dir: str = "data/sample") -> str:
    """
    Create sample synthetic data for testing.
    
    Args:
        output_dir: Directory to save sample data
        
    Returns:
        Path to sample data directory
    """
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic 6-band image
    np.random.seed(42)
    height, width = 512, 512
    n_bands = 6
    
    # Base water reflectance
    image = np.random.uniform(0.02, 0.1, (n_bands, height, width)).astype(np.float32)
    
    # Add some "debris" patches
    for _ in range(5):
        y = np.random.randint(50, height - 50)
        x = np.random.randint(50, width - 50)
        size = np.random.randint(10, 30)
        
        # Debris has higher reflectance in certain bands
        image[:, y:y+size, x:x+size] += np.random.uniform(0.1, 0.3)
    
    image = np.clip(image, 0, 1)
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    # Mark debris locations
    mask[image[3] > 0.25] = 1  # Use NIR band threshold
    
    # Save image
    image_path = output_dir / "sample_scene.tif"
    transform = from_bounds(-122.5, 37.5, -122.0, 38.0, width, height)
    
    profile = {
        "driver": "GTiff",
        "dtype": np.float32,
        "width": width,
        "height": height,
        "count": n_bands,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
    }
    
    with rasterio.open(image_path, "w", **profile) as dst:
        dst.write(image)
    
    # Save mask
    mask_path = output_dir / "sample_mask.tif"
    profile.update({"count": 1, "dtype": np.uint8})
    
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
    
    print(f"Sample data created at: {output_dir}")
    return str(output_dir)
