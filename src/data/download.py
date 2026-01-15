"""
Data download utilities for MARIDA dataset and Sentinel-2 imagery.

Note: The MARIDA dataset must be downloaded manually from GitHub
due to access restrictions. This module provides helper functions
for verification and sample data creation.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np


def create_sample_data(output_dir: str = "data/sample") -> str:
    """
    Create sample synthetic data for testing the pipeline.
    
    This creates realistic-looking synthetic Sentinel-2 imagery
    and corresponding labels for testing without the full MARIDA dataset.
    
    Args:
        output_dir: Directory to save sample data
        
    Returns:
        Path to sample data directory
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    
    output_dir = Path(output_dir)
    patches_dir = output_dir / "patches" / "S2_SAMPLE_SCENE"
    splits_dir = output_dir / "splits"
    
    patches_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple sample patches
    np.random.seed(42)
    num_patches = 10
    height, width = 256, 256
    n_bands = 12  # Sentinel-2 has 12 bands in MARIDA
    
    patch_ids = []
    
    for i in range(num_patches):
        patch_id = f"S2_SAMPLE_SCENE_{i}"
        patch_ids.append(patch_id)
        
        # Create synthetic 12-band image (realistic water/debris values)
        # Water has low reflectance, debris has higher reflectance
        image = np.random.uniform(0.01, 0.08, (n_bands, height, width)).astype(np.float32)
        
        # Add some "debris" patches with higher reflectance
        mask = np.zeros((height, width), dtype=np.uint8)
        
        num_debris = np.random.randint(0, 5)
        for _ in range(num_debris):
            y = np.random.randint(20, height - 40)
            x = np.random.randint(20, width - 40)
            size_h = np.random.randint(10, 40)
            size_w = np.random.randint(10, 40)
            
            # Debris has higher reflectance in visible and NIR bands
            debris_signature = np.random.uniform(0.1, 0.25, (n_bands, 1, 1))
            image[:, y:y+size_h, x:x+size_w] += debris_signature
            
            # Mark in mask (class 0 = Marine Debris in MARIDA)
            mask[y:y+size_h, x:x+size_w] = 0  # Will be converted to 1 for debris
        
        # Set non-debris areas to a background class (e.g., 6 = Marine Water)
        mask[mask == 0] = 6  # Marine Water
        
        # Now set debris regions
        for _ in range(num_debris):
            y = np.random.randint(20, height - 40)
            x = np.random.randint(20, width - 40)
            size_h = np.random.randint(10, 40)
            size_w = np.random.randint(10, 40)
            mask[y:y+size_h, x:x+size_w] = 0  # Marine Debris
        
        image = np.clip(image, 0, 1)
        
        # Create geotransform (sample coordinates)
        transform = from_bounds(
            -122.5 + i * 0.01, 37.5, -122.4 + i * 0.01, 37.6, 
            width, height
        )
        
        profile = {
            "driver": "GTiff",
            "dtype": np.float32,
            "width": width,
            "height": height,
            "count": n_bands,
            "crs": CRS.from_epsg(4326),
            "transform": transform,
        }
        
        # Save image
        image_path = patches_dir / f"{patch_id}.tif"
        with rasterio.open(image_path, "w", **profile) as dst:
            dst.write(image)
        
        # Save label (with _cl suffix per MARIDA convention)
        label_path = patches_dir / f"{patch_id}_cl.tif"
        profile.update({"count": 1, "dtype": np.uint8})
        with rasterio.open(label_path, "w", **profile) as dst:
            dst.write(mask, 1)
    
    # Create split files
    np.random.shuffle(patch_ids)
    n_train = int(0.7 * num_patches)
    n_val = int(0.15 * num_patches)
    
    train_ids = patch_ids[:n_train]
    val_ids = patch_ids[n_train:n_train + n_val]
    test_ids = patch_ids[n_train + n_val:]
    
    # Ensure at least one sample per split
    if len(train_ids) == 0:
        train_ids = patch_ids[:1]
    if len(val_ids) == 0:
        val_ids = patch_ids[:1]
    if len(test_ids) == 0:
        test_ids = patch_ids[:1]
    
    with open(splits_dir / "train.txt", "w") as f:
        f.write("\n".join(train_ids))
    with open(splits_dir / "val.txt", "w") as f:
        f.write("\n".join(val_ids))
    with open(splits_dir / "test.txt", "w") as f:
        f.write("\n".join(test_ids))
    
    # Create labels_mapping.txt
    labels_mapping = """0: Marine Debris
1: Dense Sargassum
2: Sparse Sargassum
3: Natural Organic Material
4: Ship
5: Clouds
6: Marine Water
7: Sediment-Laden Water
8: Foam
9: Turbid Water
10: Shallow Water
11: Waves
12: Cloud Shadows
13: Wakes
14: Mixed Water
"""
    with open(output_dir / "labels_mapping.txt", "w") as f:
        f.write(labels_mapping)
    
    print(f"Sample data created at: {output_dir}")
    print(f"  - {num_patches} patches")
    print(f"  - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return str(output_dir)


def download_sentinel2_scene(
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    output_dir: str = "data/raw",
    max_cloud_cover: float = 20.0,
    bands: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Download a Sentinel-2 scene from Microsoft Planetary Computer.
    
    Note: Requires additional packages: pystac-client, planetary-computer, stackstac
    
    Args:
        bbox: Bounding box (west, south, east, north) in WGS84
        date_range: Date range (start_date, end_date) as 'YYYY-MM-DD'
        output_dir: Directory to save scene
        max_cloud_cover: Maximum cloud cover percentage
        bands: List of bands to download
        
    Returns:
        Path to downloaded scene, or None if failed
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
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
        
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
        
    except ImportError as e:
        print(f"Required packages not installed: {e}")
        print("Install with: pip install pystac-client planetary-computer stackstac")
        return None
    except Exception as e:
        print(f"Error downloading scene: {e}")
        return None


def verify_marida_dataset(data_dir: str) -> bool:
    """
    Verify MARIDA dataset structure and contents.
    
    Args:
        data_dir: Path to MARIDA dataset directory
        
    Returns:
        True if dataset is valid, False otherwise
    """
    data_dir = Path(data_dir)
    
    required_items = [
        ("patches", "directory"),
        ("splits", "directory"),
    ]
    
    optional_items = [
        ("labels_mapping.txt", "file"),
        ("shapefiles", "directory"),
    ]
    
    print(f"Verifying MARIDA dataset at: {data_dir}")
    print("-" * 50)
    
    all_ok = True
    
    for name, item_type in required_items:
        path = data_dir / name
        if item_type == "directory":
            exists = path.is_dir()
        else:
            exists = path.is_file()
        
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name} ({item_type})")
        
        if not exists:
            all_ok = False
    
    for name, item_type in optional_items:
        path = data_dir / name
        if item_type == "directory":
            exists = path.is_dir()
        else:
            exists = path.is_file()
        
        status = "[OK]" if exists else "[OPTIONAL]"
        print(f"  {status} {name} ({item_type})")
    
    # Check patches content
    patches_dir = data_dir / "patches"
    if patches_dir.exists():
        image_files = list(patches_dir.glob("**/*.tif"))
        label_files = [f for f in image_files if "_cl.tif" in f.name]
        image_files = [f for f in image_files if "_cl.tif" not in f.name]
        
        print(f"\n  Image patches: {len(image_files)}")
        print(f"  Label patches: {len(label_files)}")
    
    print("-" * 50)
    print(f"Dataset valid: {all_ok}")
    
    return all_ok
