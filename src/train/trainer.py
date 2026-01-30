"""Training module for 3D reconstruction models."""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device, set_seed
from ..utils.metrics import evaluate_reconstruction


class Trainer:
    """Trainer class for 3D reconstruction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        save_every_n_epochs: int = 10,
        save_top_k: int = 3,
        monitor_metric: str = "val_chamfer_distance",
        monitor_mode: str = "min",
        patience: int = 20,
        min_delta: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        reprojection_loss_weight: float = 1.0,
        triangulation_loss_weight: float = 0.1,
        regularization_loss_weight: float = 0.01
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_epochs: Maximum number of epochs
            optimizer: Optimizer type
            scheduler: Scheduler type
            save_every_n_epochs: Save checkpoint every N epochs
            save_top_k: Keep top K checkpoints
            monitor_metric: Metric to monitor
            monitor_mode: Mode for monitoring ('min' or 'max')
            patience: Early stopping patience
            min_delta: Minimum change for early stopping
            checkpoint_dir: Checkpoint directory
            log_dir: Log directory
            mixed_precision: Use mixed precision training
            gradient_accumulation_steps: Gradient accumulation steps
            reprojection_loss_weight: Weight for reprojection loss
            triangulation_loss_weight: Weight for triangulation loss
            regularization_loss_weight: Weight for regularization loss
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device(device)
        self.max_epochs = max_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.save_top_k = save_top_k
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Loss weights
        self.reprojection_loss_weight = reprojection_loss_weight
        self.triangulation_loss_weight = triangulation_loss_weight
        self.regularization_loss_weight = regularization_loss_weight
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self._init_optimizer(learning_rate, weight_decay, optimizer)
        
        # Initialize scheduler
        self._init_scheduler(scheduler)
        
        # Initialize mixed precision scaler
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf') if monitor_mode == "min" else float('-inf')
        self.epochs_without_improvement = 0
        self.checkpoint_history = []
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _init_optimizer(self, learning_rate: float, weight_decay: float, optimizer: str):
        """Initialize optimizer."""
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    
    def _init_scheduler(self, scheduler: str):
        """Initialize scheduler."""
        if scheduler.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs
            )
        elif scheduler.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler.lower() == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.monitor_mode, patience=10
            )
        else:
            self.scheduler = None
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            
        Returns:
            torch.Tensor: Total loss
        """
        total_loss = 0.0
        
        # Reprojection loss
        if "points_2d" in batch and "points_3d" in outputs:
            reprojection_loss = self._compute_reprojection_loss(
                outputs["points_3d"],
                batch["points_2d"],
                batch["camera_poses"],
                outputs["camera_matrix"]
            )
            total_loss += self.reprojection_loss_weight * reprojection_loss
        
        # Triangulation loss (if ground truth available)
        if "points_3d" in batch and "points_3d" in outputs:
            triangulation_loss = self._compute_triangulation_loss(
                outputs["points_3d"],
                batch["points_3d"]
            )
            total_loss += self.triangulation_loss_weight * triangulation_loss
        
        # Regularization loss
        regularization_loss = self._compute_regularization_loss()
        total_loss += self.regularization_loss_weight * regularization_loss
        
        return total_loss
    
    def _compute_reprojection_loss(
        self,
        points_3d: torch.Tensor,
        points_2d: torch.Tensor,
        camera_poses: torch.Tensor,
        camera_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute reprojection loss."""
        # This is a simplified version - in practice, you'd implement proper projection
        # For now, return a dummy loss
        return torch.tensor(0.0, device=self.device)
    
    def _compute_triangulation_loss(
        self,
        points_3d_pred: torch.Tensor,
        points_3d_gt: torch.Tensor
    ) -> torch.Tensor:
        """Compute triangulation loss."""
        if len(points_3d_pred) == 0 or len(points_3d_gt) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Chamfer distance loss
        from ..utils.metrics import chamfer_distance
        loss = chamfer_distance(
            points_3d_pred.cpu().numpy(),
            points_3d_gt.cpu().numpy()
        )
        return torch.tensor(loss, device=self.device)
    
    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss."""
        # L2 regularization
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        return l2_reg
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch["images"])
                    loss = self.compute_loss(batch, outputs)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(batch["images"])
                loss = self.compute_loss(batch, outputs)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return {"train_loss": total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch["images"])
                loss = self.compute_loss(batch, outputs)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Compute metrics
                if "points_3d" in batch and "points_3d" in outputs:
                    metrics = evaluate_reconstruction(
                        outputs["points_3d"].cpu().numpy(),
                        batch["points_3d"].cpu().numpy()
                    )
                    
                    for key, value in metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {"val_loss": total_loss / num_batches}
        for key, values in all_metrics.items():
            avg_metrics[f"val_{key}"] = sum(values) / len(values)
        
        return avg_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "best_metric": self.best_metric
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
        
        # Save epoch checkpoint
        if self.current_epoch % self.save_every_n_epochs == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{self.current_epoch}.pt")
            torch.save(checkpoint, epoch_path)
        
        # Manage checkpoint history
        self.checkpoint_history.append(checkpoint_path)
        if len(self.checkpoint_history) > self.save_top_k:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint) and "latest" not in old_checkpoint:
                os.remove(old_checkpoint)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(all_metrics.get(self.monitor_metric, 0))
                else:
                    self.scheduler.step()
            
            # Check for improvement
            current_metric = all_metrics.get(self.monitor_metric, float('inf'))
            is_best = False
            
            if self.monitor_mode == "min":
                if current_metric < self.best_metric - self.min_delta:
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                    is_best = True
                else:
                    self.epochs_without_improvement += 1
            else:
                if current_metric > self.best_metric + self.min_delta:
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                    is_best = True
                else:
                    self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(all_metrics, is_best)
            
            # Print metrics
            print(f"Epoch {epoch}: {all_metrics}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print("Training completed!")
