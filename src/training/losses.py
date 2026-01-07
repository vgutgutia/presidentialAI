"""
Loss functions for marine debris segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Dice Loss is particularly useful for imbalanced datasets where
    the positive class (debris) is rare.
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            logits: Model output logits (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Dice loss value
        """
        num_classes = logits.shape[1]
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)  # (B, C, H*W)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Compute intersection and union per class
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        # Compute Dice coefficient per class
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes (excluding background optionally)
        dice_loss = 1.0 - dice.mean(dim=1)  # Average over classes
        
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard ones.
    
    Args:
        alpha: Class balancing factor
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Reduction method
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss.
    
    This combination is effective for segmentation tasks with
    class imbalance, as CE provides stable gradients while Dice
    directly optimizes the evaluation metric.
    
    Args:
        ce_weight: Weight for cross-entropy loss
        dice_weight: Weight for dice loss
        class_weights: Optional class weights for CE loss
    """
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights)
        else:
            self.class_weights = None
        
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model output (B, C, H, W)
            targets: Ground truth (B, H, W)
            
        Returns:
            Combined loss value
        """
        # Move class weights to correct device if needed
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None
        
        # Cross-entropy loss
        ce = F.cross_entropy(logits, targets, weight=weight)
        
        # Dice loss
        dice = self.dice_loss(logits, targets)
        
        # Combined
        total = self.ce_weight * ce + self.dice_weight * dice
        
        return total


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss to improve edge detection.
    
    Adds extra penalty for misclassifications near debris boundaries.
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
        # Sobel kernels for boundary detection
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y", 
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary-aware loss."""
        # Get predictions
        probs = F.softmax(logits, dim=1)[:, 1:2]  # Debris probability
        
        # Get target boundaries
        targets_float = targets.unsqueeze(1).float()
        
        # Compute gradients (boundaries)
        grad_x = F.conv2d(targets_float, self.sobel_x, padding=1)
        grad_y = F.conv2d(targets_float, self.sobel_y, padding=1)
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary = (boundary > 0).float()
        
        # Compute boundary loss (MSE weighted by boundary)
        diff = (probs - targets_float) ** 2
        boundary_loss = (diff * boundary).sum() / (boundary.sum() + 1e-8)
        
        return self.weight * boundary_loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss configuration dict
        
    Returns:
        Loss function module
    """
    loss_type = config.get("type", "combined")
    
    if loss_type == "ce":
        class_weights = config.get("class_weights")
        if class_weights:
            weight = torch.tensor(class_weights)
            return nn.CrossEntropyLoss(weight=weight)
        return nn.CrossEntropyLoss()
    
    elif loss_type == "dice":
        return DiceLoss()
    
    elif loss_type == "focal":
        return FocalLoss(
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0),
        )
    
    elif loss_type == "combined":
        return CombinedLoss(
            ce_weight=config.get("ce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            class_weights=config.get("class_weights"),
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
