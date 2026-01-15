"""
PyTorch Dataset classes for marine debris detection.

MARIDA Dataset Structure:
-------------------------
data/marida/
├── labels_mapping.txt
├── patches/
│   └── S2_DATE_ROI/                    # Scene folder
│       ├── S2_DATE_ROI_0.tif           # Image patch (stacked bands)
│       ├── S2_DATE_ROI_0_cl.tif        # Label patch (_cl = classification)
│       ├── S2_DATE_ROI_1.tif
│       ├── S2_DATE_ROI_1_cl.tif
│       └── ...
├── shapefiles/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

from src.data.preprocessing import normalize_bands


class MaridaDataset(Dataset):
    """
    PyTorch Dataset for the MARIDA (Marine Debris Archive) dataset.
    
    MARIDA stores patches as:
    - Image: S2_DATE_ROI_X.tif (stacked multi-band GeoTIFF)
    - Label: S2_DATE_ROI_X_cl.tif (single-band classification mask)
    
    The splits/*.txt files contain patch identifiers like "S2_DATE_ROI_X"
    
    Args:
        root_dir: Path to MARIDA dataset root directory
        split: Dataset split ('train', 'val', 'test')
        transform: Optional albumentations transform
        binary: If True, convert to binary classification (debris vs non-debris)
        normalization: Dict with 'mean' and 'std' for normalization
    """
    
    # MARIDA class mapping (from labels_mapping.txt)
    # Class 1 is Marine Debris - this is what we want to detect
    CLASS_MAPPING = {
        1: "Marine Debris",
        2: "Dense Sargassum", 
        3: "Sparse Sargassum",
        4: "Natural Organic Material",
        5: "Ship",
        6: "Clouds",
        7: "Marine Water",
        8: "Sediment-Laden Water",
        9: "Foam",
        10: "Turbid Water",
        11: "Shallow Water",
        12: "Waves",
        13: "Cloud Shadows",
        14: "Wakes",
        15: "Mixed Water",
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        binary: bool = True,
        normalization: Optional[Dict[str, List[float]]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.binary = binary
        self.normalization = normalization
        
        # Paths
        self.patches_dir = self.root_dir / "patches"
        self.splits_dir = self.root_dir / "splits"
        
        # Load patch list from split file
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            print(f"[WARNING] No samples found for split '{split}'")
            print(f"  Checked: {self.splits_dir / f'{split}.txt'}")
            print(f"  Patches dir: {self.patches_dir}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample paths from split file."""
        samples = []
        
        split_file = self.splits_dir / f"{self.split}.txt"
        
        if not split_file.exists():
            print(f"[WARNING] Split file not found: {split_file}")
            return self._load_all_samples_fallback()
        
        # Read patch IDs from split file
        with open(split_file, "r") as f:
            patch_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Loading {self.split} split: {len(patch_ids)} patch IDs")
        
        for patch_id in patch_ids:
            # patch_id format: "S2_DATE_ROI_X" 
            # Need to find the scene folder and the specific patch
            
            # The scene name is everything except the last _X part
            parts = patch_id.rsplit("_", 1)
            if len(parts) == 2:
                scene_name = parts[0]  # e.g., "S2_DATE_ROI"
                patch_num = parts[1]    # e.g., "0"
            else:
                # Fallback: try using whole name as scene
                scene_name = patch_id
                patch_num = None
            
            # Look for the scene folder
            scene_dir = self.patches_dir / scene_name
            
            if scene_dir.exists():
                # Find image and label files
                if patch_num is not None:
                    image_path = scene_dir / f"{patch_id}.tif"
                    label_path = scene_dir / f"{patch_id}_cl.tif"
                else:
                    # Search for matching files
                    image_path = None
                    label_path = None
                    for f in scene_dir.glob("*.tif"):
                        if "_cl" not in f.name:
                            image_path = f
                        elif "_cl" in f.name:
                            label_path = f
                
                if image_path and image_path.exists():
                    samples.append({
                        "id": patch_id,
                        "image_path": str(image_path),
                        "label_path": str(label_path) if label_path and label_path.exists() else None,
                    })
            else:
                # Maybe the patch_id directly refers to a file pattern
                # Search all scene folders
                found = False
                for scene_folder in self.patches_dir.iterdir():
                    if scene_folder.is_dir():
                        image_path = scene_folder / f"{patch_id}.tif"
                        label_path = scene_folder / f"{patch_id}_cl.tif"
                        
                        if image_path.exists():
                            samples.append({
                                "id": patch_id,
                                "image_path": str(image_path),
                                "label_path": str(label_path) if label_path.exists() else None,
                            })
                            found = True
                            break
                
                if not found:
                    # Try one more pattern: patch_id might be the full scene name
                    # and we need to get all patches from it
                    potential_scene = self.patches_dir / patch_id
                    if potential_scene.exists():
                        for tif_file in sorted(potential_scene.glob("*.tif")):
                            if "_cl" not in tif_file.name:
                                label_file = tif_file.parent / f"{tif_file.stem}_cl.tif"
                                samples.append({
                                    "id": tif_file.stem,
                                    "image_path": str(tif_file),
                                    "label_path": str(label_file) if label_file.exists() else None,
                                })
        
        print(f"  Found {len(samples)} valid samples")
        return samples
    
    def _load_all_samples_fallback(self) -> List[Dict[str, Any]]:
        """Fallback: load all patches from directory structure."""
        samples = []
        
        if not self.patches_dir.exists():
            print(f"[ERROR] Patches directory not found: {self.patches_dir}")
            return samples
        
        # Iterate through scene folders
        for scene_dir in sorted(self.patches_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            # Find all image patches (files without _cl suffix)
            for tif_file in sorted(scene_dir.glob("*.tif")):
                if "_cl" in tif_file.name:
                    continue  # Skip label files
                
                label_file = tif_file.parent / f"{tif_file.stem}_cl.tif"
                
                samples.append({
                    "id": tif_file.stem,
                    "image_path": str(tif_file),
                    "label_path": str(label_file) if label_file.exists() else None,
                })
        
        # Split samples
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        
        n_train = int(0.7 * len(samples))
        n_val = int(0.15 * len(samples))
        
        if self.split == "train":
            indices = indices[:n_train]
        elif self.split == "val":
            indices = indices[n_train:n_train + n_val]
        else:
            indices = indices[n_train + n_val:]
        
        samples = [samples[i] for i in indices]
        print(f"  Fallback split: {len(samples)} samples for {self.split}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image (stacked multi-band GeoTIFF)
        with rasterio.open(sample["image_path"]) as src:
            image = src.read().astype(np.float32)  # Shape: (C, H, W)
            n_bands = src.count
        
        # Load label
        if sample["label_path"] and Path(sample["label_path"]).exists():
            with rasterio.open(sample["label_path"]) as src:
                mask = src.read(1).astype(np.int64)  # Shape: (H, W)
        else:
            # No label available
            mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.int64)
        
        # Normalize image
        if self.normalization:
            # Only normalize as many bands as we have stats for
            n_stats = len(self.normalization["mean"])
            if n_bands <= n_stats:
                image = normalize_bands(
                    image,
                    mean=self.normalization["mean"][:n_bands],
                    std=self.normalization["std"][:n_bands]
                )
            else:
                # Normalize first n_stats bands
                image[:n_stats] = normalize_bands(
                    image[:n_stats],
                    mean=self.normalization["mean"],
                    std=self.normalization["std"]
                )
        
        # Convert to binary if specified
        if self.binary:
            # Class 1 is Marine Debris in MARIDA
            mask = (mask == 1).astype(np.int64)
        
        # Convert image to HWC for transforms
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        
        # Apply transforms (albumentations expects HWC)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert to PyTorch tensors
        if isinstance(image, np.ndarray):
            image = np.transpose(image, (2, 0, 1))  # Back to (C, H, W)
            image = torch.from_numpy(image.copy()).float()
            mask = torch.from_numpy(mask.copy()).long()
        else:
            # Already tensor from transform
            if image.ndim == 3 and image.shape[0] != n_bands:
                image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            image = image.float()
            mask = mask.long()
        
        return {
            "image": image,
            "mask": mask,
            "id": sample["id"],
        }


class Sentinel2Dataset(Dataset):
    """
    Dataset for inference on Sentinel-2 scenes.
    
    Handles tiling of large scenes into overlapping patches for inference.
    
    Args:
        scene_path: Path to Sentinel-2 GeoTIFF file
        tile_size: Size of tiles to extract
        overlap: Overlap between tiles
        normalization: Dict with 'mean' and 'std' for normalization
    """
    
    def __init__(
        self,
        scene_path: str,
        tile_size: int = 512,
        overlap: int = 128,
        normalization: Optional[Dict[str, List[float]]] = None,
    ):
        self.scene_path = Path(scene_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.normalization = normalization
        
        # Read scene metadata
        with rasterio.open(self.scene_path) as src:
            self.height = src.height
            self.width = src.width
            self.crs = src.crs
            self.transform = src.transform
            self.n_bands = src.count
        
        # Generate tile coordinates
        self.tiles = self._generate_tiles()
    
    def _generate_tiles(self) -> List[Tuple[int, int, int, int]]:
        """Generate list of tile coordinates (row, col, height, width)."""
        tiles = []
        stride = self.tile_size - self.overlap
        
        for row in range(0, self.height, stride):
            for col in range(0, self.width, stride):
                h = min(self.tile_size, self.height - row)
                w = min(self.tile_size, self.width - col)
                tiles.append((row, col, h, w))
        
        return tiles
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row, col, h, w = self.tiles[idx]
        
        # Read tile
        with rasterio.open(self.scene_path) as src:
            window = rasterio.windows.Window(col, row, w, h)
            image = src.read(window=window).astype(np.float32)
        
        # Pad if necessary
        if h < self.tile_size or w < self.tile_size:
            padded = np.zeros((image.shape[0], self.tile_size, self.tile_size), dtype=np.float32)
            padded[:, :h, :w] = image
            image = padded
        
        # Normalize
        if self.normalization:
            n_stats = len(self.normalization["mean"])
            n_bands = min(image.shape[0], n_stats)
            image[:n_bands] = normalize_bands(
                image[:n_bands],
                mean=self.normalization["mean"][:n_bands],
                std=self.normalization["std"][:n_bands]
            )
        
        return {
            "image": torch.from_numpy(image).float(),
            "coords": (row, col, h, w),
            "idx": idx,
        }
    
    def get_scene_info(self) -> Dict[str, Any]:
        """Get scene metadata for reconstruction."""
        return {
            "height": self.height,
            "width": self.width,
            "crs": self.crs,
            "transform": self.transform,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "n_tiles": len(self.tiles),
            "n_bands": self.n_bands,
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    bands: List[str] = None,
    normalization: Dict[str, List[float]] = None,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    patch_size: int = 256,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to MARIDA dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        bands: List of bands to use (not used - MARIDA has stacked bands)
        normalization: Normalization statistics
        transform_train: Augmentation transforms for training
        transform_val: Transforms for validation/test
        patch_size: Not used (MARIDA patches are pre-sized)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MaridaDataset(
        root_dir=data_dir,
        split="train",
        transform=transform_train,
        normalization=normalization,
    )
    
    val_dataset = MaridaDataset(
        root_dir=data_dir,
        split="val",
        transform=transform_val,
        normalization=normalization,
    )
    
    test_dataset = MaridaDataset(
        root_dir=data_dir,
        split="test",
        transform=transform_val,
        normalization=normalization,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if len(train_dataset) > batch_size else False,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
