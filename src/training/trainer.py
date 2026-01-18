"""
Training loop and utilities for marine debris detection.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics, MetricTracker


class Trainer:
    """
    Trainer class for marine debris segmentation models.
    
    Args:
        model: PyTorch model to train
        config: Training configuration dictionary
        device: Device to train on ('cuda', 'mps', 'cpu')
        output_dir: Directory to save outputs
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
        output_dir: str = "outputs",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup loss function
        loss_config = config.get("loss", {"type": "combined"})
        self.criterion = get_loss_function(loss_config)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler (will be created in train())
        self.scheduler = None
        
        # Setup tensorboard
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(
                log_dir=str(self.logs_dir / "tensorboard" / datetime.now().strftime("%Y%m%d_%H%M%S"))
            )
        else:
            self.writer = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.history = {"train": [], "val": []}
        
        # Number of classes
        self.num_classes = config.get("num_classes", 2)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        
        if optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "cosine").lower()
        
        if scheduler_name == "cosine":
            warmup_epochs = self.config.get("warmup_epochs", 5)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.config.get("min_lr", 1e-6),
            )
        elif scheduler_name == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.get("learning_rate", 1e-4),
                total_steps=num_training_steps,
            )
        elif scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )
        else:
            self.scheduler = None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        num_training_steps = epochs * len(train_loader)
        self._create_scheduler(num_training_steps)
        
        # Early stopping
        patience = self.config.get("early_stopping_patience", 20)
        no_improve_count = 0
        
        print(f"Starting training for {epochs} epochs")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            self.history["train"].append(train_metrics)
            
            # Validation epoch
            val_metrics = self._validate_epoch(val_loader, epoch)
            self.history["val"].append(val_metrics)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["iou_mean"])
                else:
                    pass  # Step per batch for other schedulers
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou_mean']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | IoU: {val_metrics['iou_mean']:.4f}")
            
            # Check for improvement
            current_metric = val_metrics["iou_mean"]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                no_improve_count = 0
                self._save_checkpoint("best_model.pth")
                print(f"  New best model! IoU: {self.best_metric:.4f}")
            else:
                no_improve_count += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get("save_every", 10) == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break
        
        # Save final model
        self._save_checkpoint("final_model.pth")
        
        return self.history
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        metric_tracker = MetricTracker(num_classes=self.num_classes)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                outputs = outputs.get("logits", outputs.get("out", list(outputs.values())[0]))
            
            # Resize outputs if needed
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            
            # Compute loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (CRITICAL for preventing NaN weights)
            grad_clip_val = self.config.get("gradient_clip_val", self.config.get("grad_clip", 1.0))
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    grad_clip_val,
                )
            
            self.optimizer.step()
            
            # Update scheduler (per step for some schedulers)
            if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                predictions = outputs.argmax(dim=1)
                metric_tracker.update(outputs, masks)
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
            
            self.global_step += 1
            
            # Log to tensorboard
            if self.writer is not None and self.global_step % 100 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        
        # Compute epoch metrics
        metrics = metric_tracker.compute()
        metrics["loss"] = total_loss / len(dataloader)
        
        return metrics
    
    def _validate_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        
        total_loss = 0.0
        metric_tracker = MetricTracker(num_classes=self.num_classes)
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")
            
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    outputs = outputs.get("logits", outputs.get("out", list(outputs.values())[0]))
                
                # Resize outputs if needed
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = nn.functional.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                
                # Compute loss
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Track metrics
                metric_tracker.update(outputs, masks)
        
        # Compute epoch metrics
        metrics = metric_tracker.compute()
        metrics["loss"] = total_loss / len(dataloader)
        
        # Log to tensorboard
        if self.writer is not None:
            self.writer.add_scalar("val/loss", metrics["loss"], epoch)
            self.writer.add_scalar("val/iou_mean", metrics["iou_mean"], epoch)
        
        return metrics
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        path = self.models_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", 0.0)
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")


def create_augmentations(config: Dict[str, Any]):
    """
    Create albumentations augmentation pipeline.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Albumentations Compose object or None
    """
    try:
        import albumentations as A
    except ImportError:
        print("[WARNING] albumentations not installed, skipping augmentations")
        return None
    
    if not config.get("enabled", True):
        return None
    
    transforms = []
    
    # Geometric transforms (safe for all image types)
    if config.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    
    if config.get("vertical_flip", True):
        transforms.append(A.VerticalFlip(p=0.5))
    
    if config.get("rotate", True):
        transforms.append(
            A.RandomRotate90(p=0.5)
        )
    
    # Note: Skipping color/noise augmentations that cause OpenCV issues
    # with multi-channel float images. The geometric transforms above
    # are sufficient for satellite imagery.
    
    if len(transforms) == 0:
        return None
    
    return A.Compose(transforms)
