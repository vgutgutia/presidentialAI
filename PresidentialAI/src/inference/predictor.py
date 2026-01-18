"""
Inference pipeline for marine debris detection.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm

from src.models.segformer import load_model
from src.data.preprocessing import normalize_bands, stitch_tiles


class MarineDebrisPredictor:
    """
    Marine debris detection inference pipeline.
    
    Handles loading model, processing satellite imagery, and generating
    georeferenced outputs (heatmaps, hotspot polygons, CSV reports).
    
    Args:
        model_path: Path to trained model checkpoint
        config: Model and inference configuration
        device: Device to run inference on
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.config = config or self._default_config()
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded on {device}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "model": {
                "backbone": "mit_b2",
                "num_classes": 2,
                "in_channels": 6,
            },
            "data": {
                "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
                "normalization": {
                    "mean": [0.0582, 0.0556, 0.0480, 0.1011, 0.1257, 0.0902],
                    "std": [0.0276, 0.0267, 0.0308, 0.0522, 0.0560, 0.0479],
                },
            },
            "inference": {
                "tile_size": 512,
                "overlap": 128,
                "batch_size": 4,
                "confidence_threshold": 0.5,
                "min_area_pixels": 100,
                "min_area_m2": 10000,
            },
        }
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        model_config = self.config.get("model", {})
        
        # Detect input channels from checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Get input channels from first conv layer
        for key in state_dict:
            if "patch_embed1.proj.weight" in key:
                model_config["in_channels"] = state_dict[key].shape[1]
                print(f"Detected input channels: {model_config['in_channels']}")
                break
        else:
            model_config["in_channels"] = 11  # Default for MARIDA
        
        model = load_model(model_path, model_config, device=self.device)
        return model
    
    def predict(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run prediction on a satellite image.
        
        Args:
            image_path: Path to input GeoTIFF
            output_dir: Directory to save outputs (optional)
            
        Returns:
            Dictionary containing:
                - probability_map: numpy array of debris probabilities
                - hotspots: GeoDataFrame of detected debris polygons
                - metadata: dict with CRS, transform, etc.
        """
        image_path = Path(image_path)
        
        # Load image and metadata
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            metadata = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "height": src.height,
                "width": src.width,
                "resolution": src.res,
            }
        
        print(f"Processing image: {image_path.name}")
        print(f"  Size: {metadata['width']} x {metadata['height']}")
        print(f"  Bands: {image.shape[0]}")
        
        # Normalize
        norm_config = self.config.get("data", {}).get("normalization", {})
        if norm_config:
            image = normalize_bands(
                image,
                mean=norm_config["mean"][:image.shape[0]],
                std=norm_config["std"][:image.shape[0]],
            )
        
        # Run inference
        probability_map = self._sliding_window_inference(image)
        
        # Post-process
        threshold = self.config["inference"]["confidence_threshold"]
        min_area = self.config["inference"]["min_area_pixels"]
        
        binary_mask = (probability_map > threshold).astype(np.uint8)
        binary_mask = self._apply_morphology(binary_mask)
        
        # Extract hotspots
        hotspots = self._extract_hotspots(
            probability_map,
            binary_mask,
            metadata["transform"],
            metadata["crs"],
            min_area,
        )
        
        results = {
            "probability_map": probability_map,
            "binary_mask": binary_mask,
            "hotspots": hotspots,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save outputs if directory specified
        if output_dir:
            self.save_results(results, output_dir, image_path.stem)
        
        return results
    
    def _sliding_window_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Run sliding window inference over large image.
        
        Args:
            image: Input image (C, H, W)
            
        Returns:
            Probability map (H, W)
        """
        tile_size = self.config["inference"]["tile_size"]
        overlap = self.config["inference"]["overlap"]
        batch_size = self.config["inference"]["batch_size"]
        
        _, height, width = image.shape
        stride = tile_size - overlap
        
        # Generate tile coordinates
        tiles = []
        coords = []
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                
                # Extract tile
                tile = image[:, y:y_end, x:x_end]
                
                # Pad if necessary
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    padded = np.zeros((image.shape[0], tile_size, tile_size), dtype=np.float32)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded
                
                tiles.append(tile)
                coords.append((y, x, y_end - y, x_end - x))
        
        # Process in batches
        predictions = []
        
        for i in tqdm(range(0, len(tiles), batch_size), desc="Processing tiles"):
            batch_tiles = tiles[i:i + batch_size]
            batch_tensor = torch.from_numpy(np.stack(batch_tiles)).float().to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = F.softmax(outputs, dim=1)[:, 1]  # Debris probability
                predictions.extend(probs.cpu().numpy())
        
        # Stitch tiles
        tile_predictions = [(pred, coord) for pred, coord in zip(predictions, coords)]
        probability_map = stitch_tiles(tile_predictions, (height, width), tile_size, overlap)
        
        return probability_map
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up predictions."""
        if not self.config["inference"].get("apply_morphology", True):
            return mask
        
        import cv2
        
        kernel_size = self.config["inference"].get("kernel_size", 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _extract_hotspots(
        self,
        probability_map: np.ndarray,
        binary_mask: np.ndarray,
        transform: Affine,
        crs: Any,
        min_area_pixels: int,
    ) -> gpd.GeoDataFrame:
        """
        Extract debris hotspot polygons from prediction.
        
        Args:
            probability_map: Debris probability values
            binary_mask: Binary debris mask
            transform: Geospatial transform
            crs: Coordinate reference system
            min_area_pixels: Minimum area threshold
            
        Returns:
            GeoDataFrame with hotspot polygons
        """
        features = []
        
        # Extract shapes from binary mask
        for geom, value in shapes(binary_mask.astype(np.int16), transform=transform):
            if value == 1:  # Debris
                poly = shape(geom)
                
                # Calculate area in pixels
                area_pixels = poly.area / (transform.a * transform.e * -1)
                
                if area_pixels >= min_area_pixels:
                    # Calculate mean confidence within polygon
                    # (simplified - use bounds for speed)
                    bounds = poly.bounds
                    
                    features.append({
                        "geometry": poly,
                        "area_m2": poly.area,
                        "confidence": float(probability_map.mean()),  # Simplified
                        "centroid_lat": poly.centroid.y,
                        "centroid_lon": poly.centroid.x,
                    })
        
        # Create GeoDataFrame
        if features:
            gdf = gpd.GeoDataFrame(features, crs=crs)
            
            # Sort by confidence and add rank
            gdf = gdf.sort_values("confidence", ascending=False).reset_index(drop=True)
            gdf["rank"] = gdf.index + 1
            
            return gdf
        else:
            return gpd.GeoDataFrame(
                columns=["geometry", "area_m2", "confidence", "centroid_lat", "centroid_lon", "rank"],
                crs=crs,
            )
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        scene_name: str,
    ):
        """
        Save prediction results to files.
        
        Args:
            results: Prediction results dictionary
            output_dir: Output directory
            scene_name: Base name for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = results["metadata"]
        
        # Save probability heatmap as GeoTIFF
        heatmap_path = output_dir / f"{scene_name}_heatmap.tif"
        
        profile = {
            "driver": "GTiff",
            "dtype": np.float32,
            "width": metadata["width"],
            "height": metadata["height"],
            "count": 1,
            "crs": metadata["crs"],
            "transform": metadata["transform"],
            "compress": "lzw",
        }
        
        with rasterio.open(heatmap_path, "w", **profile) as dst:
            dst.write(results["probability_map"].astype(np.float32), 1)
        
        print(f"Saved heatmap: {heatmap_path}")
        
        # Save hotspots as GeoJSON
        hotspots = results["hotspots"]
        
        if len(hotspots) > 0:
            geojson_path = output_dir / f"{scene_name}_hotspots.geojson"
            hotspots.to_file(geojson_path, driver="GeoJSON")
            print(f"Saved hotspots: {geojson_path}")
            
            # Save CSV summary
            csv_path = output_dir / f"{scene_name}_hotspots.csv"
            hotspots_df = hotspots.drop(columns=["geometry"])
            hotspots_df["timestamp"] = results["timestamp"]
            hotspots_df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")
        else:
            print("No hotspots detected above threshold")
        
        # Save binary mask
        mask_path = output_dir / f"{scene_name}_mask.tif"
        
        profile["dtype"] = np.uint8
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(results["binary_mask"].astype(np.uint8), 1)
        
        print(f"Saved binary mask: {mask_path}")


def batch_predict(
    predictor: MarineDebrisPredictor,
    input_dir: str,
    output_dir: str,
    pattern: str = "*.tif",
) -> List[Dict[str, Any]]:
    """
    Run batch prediction on multiple scenes.
    
    Args:
        predictor: MarineDebrisPredictor instance
        input_dir: Directory containing input images
        output_dir: Directory for outputs
        pattern: Glob pattern for input files
        
    Returns:
        List of prediction results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(input_dir.glob(pattern))
    
    if not input_files:
        print(f"No files found matching {pattern} in {input_dir}")
        return []
    
    print(f"Processing {len(input_files)} files...")
    
    all_results = []
    
    for filepath in tqdm(input_files, desc="Batch processing"):
        try:
            results = predictor.predict(str(filepath), str(output_dir))
            all_results.append({
                "file": filepath.name,
                "n_hotspots": len(results["hotspots"]),
                "status": "success",
            })
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            all_results.append({
                "file": filepath.name,
                "status": "error",
                "error": str(e),
            })
    
    return all_results
