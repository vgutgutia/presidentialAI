"""
Evaluation metrics for segmentation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class IoU:
    """
    Intersection over Union (IoU) metric.
    
    Also known as Jaccard Index.
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update statistics with new batch.
        
        Args:
            predictions: Predicted class labels (N, H, W) or (H, W)
            targets: Ground truth labels (N, H, W) or (H, W)
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Ignore specified index
        valid = targets != self.ignore_index
        predictions = predictions[valid]
        targets = targets[valid]
        
        for cls in range(self.num_classes):
            pred_mask = predictions == cls
            target_mask = targets == cls
            
            self.intersection[cls] += (pred_mask & target_mask).sum()
            self.union[cls] += (pred_mask | target_mask).sum()
    
    def compute(self) -> Dict[str, float]:
        """Compute IoU metrics."""
        iou = self.intersection / (self.union + 1e-8)
        
        results = {
            f"iou_class_{i}": iou[i] for i in range(self.num_classes)
        }
        results["iou_mean"] = iou.mean()
        
        # For binary, report debris IoU specifically
        if self.num_classes == 2:
            results["iou_debris"] = iou[1]
        
        return results


class DiceScore:
    """
    Dice coefficient / F1 Score metric.
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.tp = np.zeros(self.num_classes)  # True positives
        self.fp = np.zeros(self.num_classes)  # False positives
        self.fn = np.zeros(self.num_classes)  # False negatives
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """Update statistics with new batch."""
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Ignore specified index
        valid = targets != self.ignore_index
        predictions = predictions[valid]
        targets = targets[valid]
        
        for cls in range(self.num_classes):
            pred_mask = predictions == cls
            target_mask = targets == cls
            
            self.tp[cls] += (pred_mask & target_mask).sum()
            self.fp[cls] += (pred_mask & ~target_mask).sum()
            self.fn[cls] += (~pred_mask & target_mask).sum()
    
    def compute(self) -> Dict[str, float]:
        """Compute Dice/F1 metrics."""
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-8)
        
        results = {}
        
        for i in range(self.num_classes):
            results[f"dice_class_{i}"] = dice[i]
            results[f"precision_class_{i}"] = precision[i]
            results[f"recall_class_{i}"] = recall[i]
        
        results["dice_mean"] = dice.mean()
        results["precision_mean"] = precision.mean()
        results["recall_mean"] = recall.mean()
        
        # For binary, report debris metrics specifically
        if self.num_classes == 2:
            results["dice_debris"] = dice[1]
            results["precision_debris"] = precision[1]
            results["recall_debris"] = recall[1]
            results["f1_debris"] = dice[1]  # Dice == F1
        
        return results


class ConfusionMatrix:
    """
    Confusion matrix accumulator.
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset confusion matrix."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """Update confusion matrix."""
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        for pred, target in zip(predictions, targets):
            if 0 <= target < self.num_classes and 0 <= pred < self.num_classes:
                self.matrix[target, pred] += 1
    
    def compute(self) -> np.ndarray:
        """Return confusion matrix."""
        return self.matrix
    
    def get_normalized(self) -> np.ndarray:
        """Return row-normalized confusion matrix."""
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        return self.matrix / (row_sums + 1e-8)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics.
    
    Args:
        predictions: Model predictions (B, C, H, W) logits or (B, H, W) probs
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metric values
    """
    # Handle different prediction formats
    if predictions.dim() == 4:
        # Logits (B, C, H, W) -> class predictions
        pred_classes = predictions.argmax(dim=1)
    elif predictions.dim() == 3:
        if predictions.max() <= 1.0:
            # Probabilities -> threshold
            pred_classes = (predictions > threshold).long()
        else:
            # Already class predictions
            pred_classes = predictions.long()
    else:
        pred_classes = predictions.long()
    
    # Convert to numpy
    pred_np = pred_classes.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Initialize metrics
    iou_metric = IoU(num_classes=num_classes)
    dice_metric = DiceScore(num_classes=num_classes)
    
    # Update metrics
    iou_metric.update(pred_np, target_np)
    dice_metric.update(pred_np, target_np)
    
    # Compute and merge results
    results = {}
    results.update(iou_metric.compute())
    results.update(dice_metric.compute())
    
    return results


def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth labels
        
    Returns:
        Pixel accuracy value
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)
    
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    
    return correct / total


class MetricTracker:
    """
    Track metrics across batches and epochs.
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all trackers."""
        self.iou = IoU(num_classes=self.num_classes)
        self.dice = DiceScore(num_classes=self.num_classes)
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ):
        """Update metrics with batch results."""
        # Convert predictions to class labels
        if predictions.dim() == 4:
            pred_classes = predictions.argmax(dim=1)
        else:
            pred_classes = predictions
        
        pred_np = pred_classes.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        self.iou.update(pred_np, target_np)
        self.dice.update(pred_np, target_np)
        
        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        results = {}
        results.update(self.iou.compute())
        results.update(self.dice.compute())
        
        if self.num_batches > 0:
            results["loss"] = self.total_loss / self.num_batches
        
        return results
