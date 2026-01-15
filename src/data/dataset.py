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
    """
    
    # MARIDA class mapping
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
        target_channels: int = 11,  # Expected number of channels
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.binary = binary
        self.normalization = normalization
        self.target_channels = target_channels
        
        # Paths
        self.patches_dir = self.root_dir / "patches"
        self.splits_dir = self.root_dir / "splits"
        
        # Load and validate samples
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            print(f"[WARNING] No samples found for split '{split}'")
    
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
            sample = self._find_patch_files(patch_id)
            if sample is not None:
                samples.append(sample)
        
        print(f"  Found {len(samples)} valid samples")
        return samples
    
    def _find_patch_files(self, patch_id: str) -> Optional[Dict[str, Any]]:
        """Find image and label files for a patch ID."""
        # patch_id format: "S2_DATE_ROI_X" 
        # Scene name is everything except the last _X part
        parts = patch_id.rsplit("_", 1)
        if len(parts) == 2:
            scene_name = parts[0]
        else:
            scene_name = patch_id
        
        # Look for the scene folder
        scene_dir = self.patches_dir / scene_name
        
        if scene_dir.exists():
            image_path = scene_dir / f"{patch_id}.tif"
            label_path = scene_dir / f"{patch_id}_cl.tif"
            
            if image_path.exists():
                # Validate the image has correct number of channels
                try:
                    with rasterio.open(image_path) as src:
                        n_bands = src.count
                        if n_bands < 3:  # Skip clearly invalid images
                            return None
                except:
                    return None
                
                return {
                    "id": patch_id,
                    "image_path": str(image_path),
                    "label_path": str(label_path) if label_path.exists() else None,
                }
        
        # Search all scene folders
        for scene_folder in self.patches_dir.iterdir():
            if scene_folder.is_dir():
                image_path = scene_folder / f"{patch_id}.tif"
                label_path = scene_folder / f"{patch_id}_cl.tif"
                
                if image_path.exists():
                    try:
                        with rasterio.open(image_path) as src:
                            n_bands = src.count
                            if n_bands < 3:
                                continue
                    except:
                        continue
                    
                    return {
                        "id": patch_id,
                        "image_path": str(image_path),
                        "label_path": str(label_path) if label_path.exists() else None,
                    }
        
        return None
    
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
                    continue
                
                # Validate image
                try:
                    with rasterio.open(tif_file) as src:
                        n_bands = src.count
                        if n_bands < 3:
                            continue
                except:
                    continue
                
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
        
        # Load image
        with rasterio.open(sample["image_path"]) as src:
            image = src.read().astype(np.float32)
            n_bands = src.count
        
        # Ensure consistent number of channels
        if n_bands < self.target_channels:
            # Pad with zeros
            padded = np.zeros((self.target_channels, image.shape[1], image.shape[2]), dtype=np.float32)
            padded[:n_bands] = image
            image = padded
        elif n_bands > self.target_channels:
            # Truncate
            image = image[:self.target_channels]
        
        # Load label
        if sample["label_path"] and Path(sample["label_path"]).exists():
            with rasterio.open(sample["label_path"]) as src:
                mask = src.read(1).astype(np.int64)
        else:
            mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.int64)
        
        # Normalize
        if self.normalization:
            n_stats = min(len(self.normalization["mean"]), image.shape[0])
            image[:n_stats] = normalize_bands(
                image[:n_stats],
                mean=self.normalization["mean"][:n_stats],
                std=self.normalization["std"][:n_stats]
            )
        
        # Binary classification
        if self.binary:
            mask = (mask == 1).astype(np.int64)
        
        # Apply transforms
        if self.transform:
            image_hwc = np.transpose(image, (1, 2, 0))
            transformed = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(transformed["image"], (2, 0, 1))
            mask = transformed["mask"]
        
        # Convert to tensors
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        
        return {
            "image": image,
            "mask": mask,
            "id": sample["id"],
        }


class Sentinel2Dataset(Dataset):
    """Dataset for inference on Sentinel-2 scenes."""
    
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
        
        with rasterio.open(self.scene_path) as src:
            self.height = src.height
            self.width = src.width
            self.crs = src.crs
            self.transform = src.transform
            self.n_bands = src.count
        
        self.tiles = self._generate_tiles()
    
    def _generate_tiles(self) -> List[Tuple[int, int, int, int]]:
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
        
        with rasterio.open(self.scene_path) as src:
            window = rasterio.windows.Window(col, row, w, h)
            image = src.read(window=window).astype(np.float32)
        
        if h < self.tile_size or w < self.tile_size:
            padded = np.zeros((image.shape[0], self.tile_size, self.tile_size), dtype=np.float32)
            padded[:, :h, :w] = image
            image = padded
        
        if self.normalization:
            n_stats = min(len(self.normalization["mean"]), image.shape[0])
            image[:n_stats] = normalize_bands(
                image[:n_stats],
                mean=self.normalization["mean"][:n_stats],
                std=self.normalization["std"][:n_stats]
            )
        
        return {
            "image": torch.from_numpy(image).float(),
            "coords": (row, col, h, w),
            "idx": idx,
        }
    
    def get_scene_info(self) -> Dict[str, Any]:
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
    target_channels: int = 11,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    train_dataset = MaridaDataset(
        root_dir=data_dir,
        split="train",
        transform=transform_train,
        normalization=normalization,
        target_channels=target_channels,
    )
    
    val_dataset = MaridaDataset(
        root_dir=data_dir,
        split="val",
        transform=transform_val,
        normalization=normalization,
        target_channels=target_channels,
    )
    
    test_dataset = MaridaDataset(
        root_dir=data_dir,
        split="test",
        transform=transform_val,
        normalization=normalization,
        target_channels=target_channels,
    )
    
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
