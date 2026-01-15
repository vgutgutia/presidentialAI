"""
Training loop and utilities.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.training.losses import get_loss_function
from src.training.metrics import MetricTracker, compute_metrics


console = Console()


class Trainer:
    """
    Training loop manager for marine debris detection.
    
    Handles training, validation, checkpointing, and logging.
    
    Args:
        model: Model to train
        config: Training configuration dict
        device: Device to train on ('mps', 'cuda', 'cpu')
        output_dir: Directory for outputs
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cpu",
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loss function
        self.criterion = get_loss_function(config.get("loss", {}))
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=tuple(config.get("betas", [0.9, 0.999])),
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler(config)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = {"train": [], "val": []}
        
        # TensorBoard writer
        self.writer = None
        if config.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.logs_dir / "tensorboard")
            except ImportError:
                console.print("[yellow]TensorBoard not available[/yellow]")
    
    def _create_scheduler(self, config: Dict) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_type = config.get("scheduler", "cosine")
        epochs = config.get("epochs", 100)
        warmup_epochs = config.get("warmup_epochs", 5)
        min_lr = config.get("min_lr", 1e-6)
        
        if scheduler_type == "cosine":
            # Warmup + cosine annealing
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
            
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=min_lr,
            )
            
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
            
            return scheduler
        
        return None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.get("epochs", 100)
        
        console.print(f"\n[bold green]Starting training for {epochs} epochs[/bold green]")
        console.print(f"Device: {self.device}")
        console.print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Early stopping
        early_stop_config = self.config.get("early_stopping", {})
        patience = early_stop_config.get("patience", 20)
        min_delta = early_stop_config.get("min_delta", 0.001)
        epochs_without_improvement = 0
        
        try:
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch(train_loader, epoch)
                self.history["train"].append(train_metrics)
                
                # Validation phase
                val_metrics = self._validate_epoch(val_loader, epoch)
                self.history["val"].append(val_metrics)
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Log metrics
                self._log_epoch(epoch, train_metrics, val_metrics)
                
                # Checkpointing
                monitor_metric = val_metrics.get("iou_debris", val_metrics.get("iou_mean", 0))
                
                if monitor_metric > self.best_metric + min_delta:
                    self.best_metric = monitor_metric
                    self._save_checkpoint("best_model.pth", is_best=True)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Save periodic checkpoint
                if (epoch + 1) % self.config.get("save_every_n_epochs", 10) == 0:
                    self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
                
                # Early stopping
                if early_stop_config.get("enabled", True) and epochs_without_improvement >= patience:
                    console.print(f"\n[yellow]Early stopping triggered after {patience} epochs without improvement[/yellow]")
                    break
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted by user[/yellow]")
        
        finally:
            # Save final checkpoint
            self._save_checkpoint("final_model.pth")
            
            if self.writer:
                self.writer.close()
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        metric_tracker = MetricTracker(num_classes=self.config.get("num_classes", 2))
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                metric_tracker.update(outputs, masks, loss=loss.item())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return metric_tracker.compute()
    
    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        
        metric_tracker = MetricTracker(num_classes=self.config.get("num_classes", 2))
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            metric_tracker.update(outputs, masks, loss=loss.item())
        
        return metric_tracker.compute()
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log epoch results."""
        # Create table
        table = Table(title=f"Epoch {epoch + 1}")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="green")
        table.add_column("Val", style="yellow")
        
        # Key metrics to display
        key_metrics = ["loss", "iou_debris", "dice_debris", "precision_debris", "recall_debris"]
        
        for metric in key_metrics:
            train_val = train_metrics.get(metric, 0)
            val_val = val_metrics.get(metric, 0)
            table.add_row(metric, f"{train_val:.4f}", f"{val_val:.4f}")
        
        console.print(table)
        
        # TensorBoard logging
        if self.writer:
            for name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{name}", value, epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f"val/{name}", value, epoch)
            
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "config": self.config,
            "history": self.history,
        }
        
        path = self.models_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            console.print(f"[green]✓ Saved best model (IoU: {self.best_metric:.4f})[/green]")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric = checkpoint.get("best_metric", 0)
        self.history = checkpoint.get("history", {"train": [], "val": []})
        
        console.print(f"[green]Loaded checkpoint from epoch {self.current_epoch}[/green]")


def create_augmentations(config: Dict) -> Any:
    """
    Create data augmentation transforms.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Albumentations transform pipeline
    """
    import albumentations as A
    
    if not config.get("enabled", True):
        return None
    
    transforms = [
        A.HorizontalFlip(p=config.get("horizontal_flip", 0.5)),
        A.VerticalFlip(p=config.get("vertical_flip", 0.5)),
        A.RandomRotate90(p=config.get("rotate_90", 0.5)),
    ]
    
    if config.get("brightness_contrast", 0) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=config.get("brightness_contrast", 0.2),
                contrast_limit=config.get("brightness_contrast", 0.2),
                p=0.5,
            )
        )
    
    if config.get("gaussian_noise", 0) > 0:
        transforms.append(
            A.GaussNoise(
                std_range=(0, config.get("gaussian_noise", 0.1)),
                p=0.3,
            )
        )
    
    return A.Compose(transforms)
