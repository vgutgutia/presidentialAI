"""
Unit tests for marine debris detection system.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np


class TestDevice:
    """Test device detection utilities."""
    
    def test_get_device(self):
        """Test device auto-detection."""
        from src.utils.device import get_device
        
        device = get_device("auto")
        assert device in ["mps", "cuda", "cpu"]
    
    def test_get_device_cpu(self):
        """Test explicit CPU selection."""
        from src.utils.device import get_device
        
        device = get_device("cpu")
        assert device == "cpu"
    
    def test_device_info(self):
        """Test device info retrieval."""
        from src.utils.device import get_device_info
        
        info = get_device_info()
        assert "pytorch_version" in info
        assert "cpu_available" in info
        assert info["cpu_available"] == True


class TestConfig:
    """Test configuration utilities."""
    
    def test_default_config(self):
        """Test default configuration."""
        from src.utils.config import get_default_config
        
        config = get_default_config()
        assert "model" in config
        assert "training" in config
        assert "data" in config
        assert "inference" in config
    
    def test_merge_configs(self):
        """Test config merging."""
        from src.utils.config import merge_configs
        
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"c": 3, "d": 4}}
        
        merged = merge_configs(base, override)
        
        assert merged["a"] == 1
        assert merged["b"]["c"] == 3
        assert merged["b"]["d"] == 4


class TestPreprocessing:
    """Test preprocessing utilities."""
    
    def test_normalize_bands(self):
        """Test band normalization."""
        from src.data.preprocessing import normalize_bands
        
        # Create dummy image (6 bands, 64x64)
        image = np.random.rand(6, 64, 64).astype(np.float32) * 0.2
        
        mean = [0.1] * 6
        std = [0.05] * 6
        
        normalized = normalize_bands(image, mean, std)
        
        assert normalized.shape == image.shape
        assert normalized.dtype == np.float32
    
    def test_create_tiles(self):
        """Test tile creation."""
        from src.data.preprocessing import create_tiles
        
        image = np.random.rand(6, 256, 256).astype(np.float32)
        
        tiles = create_tiles(image, tile_size=128, overlap=32)
        
        assert len(tiles) > 0
        assert tiles[0][0].shape == (6, 128, 128)
    
    def test_stitch_tiles(self):
        """Test tile stitching."""
        from src.data.preprocessing import create_tiles, stitch_tiles
        
        # Create image and tile it
        original = np.random.rand(1, 256, 256).astype(np.float32)
        tiles = create_tiles(original, tile_size=128, overlap=32)
        
        # Stitch back
        stitched = stitch_tiles(tiles, (256, 256), tile_size=128, overlap=32)
        
        # Should be close to original (some blending at edges)
        assert stitched.shape == (256, 256)


class TestModel:
    """Test model utilities."""
    
    def test_create_model(self):
        """Test model creation."""
        from src.models.segformer import create_model
        
        config = {
            "backbone": "mit_b2",
            "num_classes": 2,
            "in_channels": 6,
            "pretrained": False,  # Don't download for tests
        }
        
        model = create_model(config, device="cpu")
        
        assert model is not None
        assert hasattr(model, "forward")
    
    def test_model_forward(self):
        """Test model forward pass."""
        from src.models.segformer import create_model
        
        config = {
            "backbone": "mit_b2",
            "num_classes": 2,
            "in_channels": 6,
            "pretrained": False,
        }
        
        model = create_model(config, device="cpu")
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 6, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 2  # Num classes
        assert output.shape[2] == 256  # Height
        assert output.shape[3] == 256  # Width


class TestLosses:
    """Test loss functions."""
    
    def test_dice_loss(self):
        """Test Dice loss computation."""
        from src.training.losses import DiceLoss
        
        loss_fn = DiceLoss()
        
        # Create dummy predictions and targets
        logits = torch.randn(2, 2, 64, 64)
        targets = torch.randint(0, 2, (2, 64, 64))
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0
        assert loss.item() <= 1
    
    def test_combined_loss(self):
        """Test combined loss computation."""
        from src.training.losses import CombinedLoss
        
        loss_fn = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
        
        logits = torch.randn(2, 2, 64, 64)
        targets = torch.randint(0, 2, (2, 64, 64))
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0
    
    def test_get_loss_function(self):
        """Test loss function factory."""
        from src.training.losses import get_loss_function
        
        # Test different loss types
        for loss_type in ["ce", "dice", "combined"]:
            config = {"type": loss_type}
            loss_fn = get_loss_function(config)
            assert loss_fn is not None


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_iou_metric(self):
        """Test IoU computation."""
        from src.training.metrics import IoU
        
        iou = IoU(num_classes=2)
        
        # Perfect predictions
        predictions = np.array([0, 0, 1, 1])
        targets = np.array([0, 0, 1, 1])
        
        iou.update(predictions, targets)
        results = iou.compute()
        
        assert results["iou_mean"] == 1.0
    
    def test_dice_score(self):
        """Test Dice score computation."""
        from src.training.metrics import DiceScore
        
        dice = DiceScore(num_classes=2)
        
        predictions = np.array([0, 0, 1, 1])
        targets = np.array([0, 0, 1, 1])
        
        dice.update(predictions, targets)
        results = dice.compute()
        
        assert results["dice_mean"] == 1.0
    
    def test_compute_metrics(self):
        """Test combined metrics computation."""
        from src.training.metrics import compute_metrics
        
        predictions = torch.tensor([[[0, 0], [1, 1]]])  # (B, H, W)
        targets = torch.tensor([[[0, 0], [1, 1]]])
        
        metrics = compute_metrics(predictions, targets, num_classes=2)
        
        assert "iou_mean" in metrics
        assert "dice_mean" in metrics


class TestDataset:
    """Test dataset classes."""
    
    def test_sentinel2_dataset_init(self):
        """Test Sentinel2Dataset initialization (without actual file)."""
        from src.data.dataset import Sentinel2Dataset
        
        # This will fail without actual file, but tests import
        with pytest.raises(Exception):
            dataset = Sentinel2Dataset("nonexistent.tif")


# Integration tests (require data)
class TestIntegration:
    """Integration tests requiring sample data."""
    
    @pytest.mark.skipif(
        not Path("data/sample").exists(),
        reason="Sample data not available"
    )
    def test_sample_data_creation(self):
        """Test sample data creation."""
        from src.data.download import create_sample_data
        
        output_dir = create_sample_data("data/sample_test")
        
        assert Path(output_dir).exists()
        assert (Path(output_dir) / "sample_scene.tif").exists()
        
        # Cleanup
        import shutil
        shutil.rmtree("data/sample_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
