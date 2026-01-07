"""
PyTorch Dataset classes for marine debris detection.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
from PIL import Image

from src.data.preprocessing import normalize_bands


class MaridaDataset(Dataset):
    """
    PyTorch Dataset for the MARIDA (Marine Debris Archive) dataset.
    
    The MARIDA dataset contains Sentinel-2 multispectral imagery with
    pixel-level semantic annotations for marine debris detection.
    
    Args:
        root_dir: Path to MARIDA dataset root directory
        split: Dataset split ('train', 'val', 'test')
        bands: List of band names to use
        transform: Optional albumentations transform
        patch_size: Size of patches to extract
        binary: If True, convert to binary classification (debris vs non-debris)
        normalization: Dict with 'mean' and 'std' for normalization
    """
    
    # MARIDA class IDs
    CLASS_NAMES = {
        0: "Marine_Water",
        1: "Marine_Debris", 
        2: "Dense_Sargassum",
        3: "Sparse_Sargassum",
        4: "Natural_Organic_Material",
        5: "Ship",
        6: "Clouds",
        7: "Sediment_Laden_Water",
        8: "Foam",
        9: "Turbid_Water",
        10: "Shallow_Water",
        11: "Waves",
        12: "Cloud_Shadows",
        13: "Wakes",
        14: "Mixed_Water",
    }
    
    # Map band names to MARIDA file indices
    BAND_MAP = {
        "B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5,
        "B7": 6, "B8": 7, "B8A": 8, "B9": 9, "B11": 10, "B12": 11
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        bands: List[str] = None,
        transform: Optional[Callable] = None,
        patch_size: int = 256,
        binary: bool = True,
        normalization: Optional[Dict[str, List[float]]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.bands = bands or ["B2", "B3", "B4", "B8", "B11", "B12"]
        self.transform = transform
        self.patch_size = patch_size
        self.binary = binary
        self.normalization = normalization
        
        # Get band indices
        self.band_indices = [self.BAND_MAP[b] for b in self.bands]
        
        # Load split file
        self.samples = self._load_split()
        
    def _load_split(self) -> List[Dict[str, Any]]:
        """Load the dataset split."""
        samples = []
        
        # Path to splits file
        splits_dir = self.root_dir / "splits"
        split_file = splits_dir / f"{self.split}.txt"
        
        # If split file doesn't exist, create from available data
        if not split_file.exists():
            samples = self._create_samples_from_directory()
        else:
            with open(split_file, "r") as f:
                patch_ids = [line.strip() for line in f.readlines()]
            
            for patch_id in patch_ids:
                # Parse patch ID to get scene and patch info
                img_path = self.root_dir / "patches" / f"{patch_id}.tif"
                mask_path = self.root_dir / "masks" / f"{patch_id}.tif"
                
                if img_path.exists() and mask_path.exists():
                    samples.append({
                        "id": patch_id,
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                    })
        
        return samples
    
    def _create_samples_from_directory(self) -> List[Dict[str, Any]]:
        """Create samples list by scanning directory."""
        samples = []
        
        patches_dir = self.root_dir / "patches"
        masks_dir = self.root_dir / "masks"
        
        if not patches_dir.exists():
            # Try alternative structure
            patches_dir = self.root_dir / "scenes"
            masks_dir = self.root_dir / "labels"
        
        if patches_dir.exists():
            for img_file in sorted(patches_dir.glob("*.tif")):
                mask_file = masks_dir / img_file.name
                
                if mask_file.exists():
                    samples.append({
                        "id": img_file.stem,
                        "image_path": str(img_file),
                        "mask_path": str(mask_file),
                    })
        
        # Split samples based on split type
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        
        n_train = int(0.7 * len(samples))
        n_val = int(0.15 * len(samples))
        
        if self.split == "train":
            indices = indices[:n_train]
        elif self.split == "val":
            indices = indices[n_train:n_train + n_val]
        else:  # test
            indices = indices[n_train + n_val:]
        
        return [samples[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        with rasterio.open(sample["image_path"]) as src:
            # Read specified bands
            image = src.read(self.band_indices).astype(np.float32)
            # Shape: (C, H, W)
        
        # Load mask
        with rasterio.open(sample["mask_path"]) as src:
            mask = src.read(1).astype(np.int64)
        
        # Normalize image
        if self.normalization:
            image = normalize_bands(
                image,
                mean=self.normalization["mean"],
                std=self.normalization["std"]
            )
        
        # Convert to binary if specified
        if self.binary:
            # Class 1 is Marine_Debris in MARIDA
            mask = (mask == 1).astype(np.int64)
        
        # Convert to HWC for transforms
        image = np.transpose(image, (1, 2, 0))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert back to CHW for PyTorch
        if isinstance(image, np.ndarray):
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).long()
        else:
            image = image.permute(2, 0, 1)
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
        bands: List of band names (assumes bands are in order in the file)
        tile_size: Size of tiles to extract
        overlap: Overlap between tiles
        normalization: Dict with 'mean' and 'std' for normalization
    """
    
    def __init__(
        self,
        scene_path: str,
        bands: List[str] = None,
        tile_size: int = 512,
        overlap: int = 128,
        normalization: Optional[Dict[str, List[float]]] = None,
    ):
        self.scene_path = Path(scene_path)
        self.bands = bands or ["B2", "B3", "B4", "B8", "B11", "B12"]
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
                # Calculate actual tile size (may be smaller at edges)
                h = min(self.tile_size, self.height - row)
                w = min(self.tile_size, self.width - col)
                
                tiles.append((row, col, h, w))
        
        return tiles
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row, col, h, w = self.tiles[idx]
        
        # Read tile from scene
        with rasterio.open(self.scene_path) as src:
            window = Window(col, row, w, h)
            
            # Read all bands or specified bands
            if self.n_bands == len(self.bands):
                image = src.read(window=window).astype(np.float32)
            else:
                # Assume first n bands are what we need
                image = src.read(
                    list(range(1, len(self.bands) + 1)),
                    window=window
                ).astype(np.float32)
        
        # Pad if necessary
        if h < self.tile_size or w < self.tile_size:
            padded = np.zeros(
                (image.shape[0], self.tile_size, self.tile_size),
                dtype=np.float32
            )
            padded[:, :h, :w] = image
            image = padded
        
        # Normalize
        if self.normalization:
            image = normalize_bands(
                image,
                mean=self.normalization["mean"],
                std=self.normalization["std"]
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
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    bands: List[str] = None,
    normalization: Dict[str, List[float]] = None,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to MARIDA dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        bands: List of bands to use
        normalization: Normalization statistics
        transform_train: Augmentation transforms for training
        transform_val: Transforms for validation/test
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MaridaDataset(
        root_dir=data_dir,
        split="train",
        bands=bands,
        transform=transform_train,
        normalization=normalization,
    )
    
    val_dataset = MaridaDataset(
        root_dir=data_dir,
        split="val",
        bands=bands,
        transform=transform_val,
        normalization=normalization,
    )
    
    test_dataset = MaridaDataset(
        root_dir=data_dir,
        split="test",
        bands=bands,
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
        drop_last=True,
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
