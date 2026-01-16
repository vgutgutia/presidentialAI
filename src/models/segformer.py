"""
SegFormer model adapted for multispectral satellite imagery.

This module provides a modified SegFormer architecture that accepts
N-band input instead of standard 3-channel RGB images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from pathlib import Path


class SegFormerMultispectral(nn.Module):
    """
    SegFormer model adapted for multispectral satellite imagery.
    
    This model modifies the standard SegFormer to accept arbitrary
    number of input channels while leveraging pretrained weights.
    
    Args:
        backbone: SegFormer backbone variant (mit_b0 to mit_b5)
        num_classes: Number of output classes
        in_channels: Number of input channels (bands)
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to custom pretrained weights
    """
    
    BACKBONE_CONFIGS = {
        "mit_b0": {"embed_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2]},
        "mit_b1": {"embed_dims": [64, 128, 320, 512], "depths": [2, 2, 2, 2]},
        "mit_b2": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 6, 3]},
        "mit_b3": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 18, 3]},
        "mit_b4": {"embed_dims": [64, 128, 320, 512], "depths": [3, 8, 27, 3]},
        "mit_b5": {"embed_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3]},
    }
    
    def __init__(
        self,
        backbone: str = "mit_b2",
        num_classes: int = 2,
        in_channels: int = 6,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Build model using segmentation_models_pytorch or transformers
        self._build_model(pretrained, pretrained_path)
    
    def _build_model(self, pretrained: bool, pretrained_path: Optional[str]):
        """Build the model architecture."""
        try:
            # Try using segmentation_models_pytorch (simpler)
            import segmentation_models_pytorch as smp
            
            self.model = smp.Unet(
                encoder_name=f"mit_{self.backbone_name[-2:]}",
                encoder_weights="imagenet" if pretrained else None,
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
            self.use_smp = True
            return
        except Exception:
            pass
        
        # Fallback to transformers implementation
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
            
            # Create config
            config = SegformerConfig(
                num_channels=self.in_channels,
                num_labels=self.num_classes,
                **self.BACKBONE_CONFIGS.get(self.backbone_name, self.BACKBONE_CONFIGS["mit_b2"])
            )
            
            if pretrained:
                # Load pretrained and adapt
                model_name = f"nvidia/segformer-{self.backbone_name[-2:]}-finetuned-ade-512-512"
                base_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
                
                # Create new model with correct input channels
                self.model = SegformerForSemanticSegmentation(config)
                
                # Copy weights, adapting first layer
                self._adapt_pretrained_weights(base_model)
            else:
                self.model = SegformerForSemanticSegmentation(config)
            
            self.use_smp = False
            return
        except Exception as e:
            print(f"Could not load transformers model: {e}")
        
        # Ultimate fallback: simple U-Net
        self._build_simple_unet()
        self.use_smp = False
    
    def _adapt_pretrained_weights(self, pretrained_model):
        """Adapt pretrained 3-channel weights to N-channel input."""
        # Get pretrained state dict
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.model.state_dict()
        
        # Find the first conv layer and adapt it
        for key in pretrained_dict:
            if key in model_dict:
                pretrained_shape = pretrained_dict[key].shape
                model_shape = model_dict[key].shape
                
                if pretrained_shape == model_shape:
                    # Same shape, copy directly
                    model_dict[key] = pretrained_dict[key]
                elif len(pretrained_shape) == 4 and pretrained_shape[1] == 3:
                    # This is likely the first conv layer
                    # Adapt 3-channel weights to N-channel
                    model_dict[key] = self._adapt_input_layer(
                        pretrained_dict[key],
                        self.in_channels
                    )
        
        self.model.load_state_dict(model_dict)
    
    def _adapt_input_layer(self, weights: torch.Tensor, target_channels: int) -> torch.Tensor:
        """
        Adapt 3-channel input weights to N-channel.
        
        Strategy: Average RGB weights and replicate, then add small random noise.
        """
        # weights shape: (out_channels, 3, kernel_h, kernel_w)
        out_channels = weights.shape[0]
        kernel_size = weights.shape[2:]
        
        # Average across input channels
        mean_weights = weights.mean(dim=1, keepdim=True)
        
        # Replicate for target channels
        new_weights = mean_weights.repeat(1, target_channels, 1, 1)
        
        # Add small random initialization for diversity
        noise = torch.randn_like(new_weights) * 0.01
        new_weights = new_weights + noise
        
        return new_weights
    
    def _build_simple_unet(self):
        """Build a simple U-Net as ultimate fallback."""
        self.model = SimpleUNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes, H, W)
        """
        if self.use_smp:
            return self.model(x)
        
        # Handle transformers output
        if hasattr(self.model, "forward"):
            outputs = self.model(x)
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Upsample to input size if needed
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            
            return logits
        
        return self.model(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make predictions with optional thresholding.
        
        Args:
            x: Input tensor
            threshold: Confidence threshold for binary classification
            
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            
            if self.num_classes == 2:
                # Binary classification - return debris probability
                return probs[:, 1]
            else:
                return probs.argmax(dim=1)


class SimpleUNet(nn.Module):
    """
    Simple U-Net implementation as fallback.
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        base_features: int = 64,
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, 2)
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, 2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, 2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, 2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        # Output
        self.out = nn.Conv2d(base_features, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


def create_model(
    config: Dict[str, Any],
    device: Optional[str] = None,
) -> nn.Module:
    """
    Create a model from configuration.
    
    Args:
        config: Model configuration dict
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    model = SegFormerMultispectral(
        backbone=config.get("backbone", "mit_b2"),
        num_classes=config.get("num_classes", 2),
        in_channels=config.get("in_channels", 6),
        pretrained=config.get("pretrained", True),
    )
    
    if device:
        model = model.to(device)
    
    return model


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: Optional[str] = None,
) -> nn.Module:
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = create_model(config, device=None)
    
    # weights_only=False needed for checkpoints with numpy arrays
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    if device:
        model = model.to(device)
    
    model.eval()
    return model