#!/usr/bin/env python3
"""Main training script for multi-view 3D reconstruction."""

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.synthetic_dataset import SyntheticMultiViewDataset
from src.models.sfm_basic import SfMBasicModel
from src.train.trainer import Trainer
from src.utils.device import get_device, set_seed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multi-view 3D reconstruction model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.device != "auto":
        config.device = args.device
    if args.seed != 42:
        config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SyntheticMultiViewDataset(
        num_images=config.data.num_images,
        image_size=tuple(config.data.image_size),
        noise_level=config.data.noise_level,
        blur_sigma=config.data.blur_sigma,
        camera_radius=config.data.camera_radius,
        camera_height=config.data.camera_height,
        look_at_center=tuple(config.data.look_at_center),
        num_objects=config.data.num_objects,
        object_scale_range=tuple(config.data.object_scale_range),
        object_position_range=tuple(config.data.object_position_range),
        split="train",
        num_samples=1000
    )
    
    val_dataset = SyntheticMultiViewDataset(
        num_images=config.data.num_images,
        image_size=tuple(config.data.image_size),
        noise_level=config.data.noise_level,
        blur_sigma=config.data.blur_sigma,
        camera_radius=config.data.camera_radius,
        camera_height=config.data.camera_height,
        look_at_center=tuple(config.data.look_at_center),
        num_objects=config.data.num_objects,
        object_scale_range=tuple(config.data.object_scale_range),
        object_position_range=tuple(config.data.object_position_range),
        split="val",
        num_samples=200
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Create model
    print("Creating model...")
    model = SfMBasicModel(
        feature_detector=config.model.feature_detector,
        matcher=config.model.matcher,
        min_matches=config.model.min_matches,
        ransac_threshold=config.model.ransac_threshold,
        confidence=config.model.confidence,
        focal_length=config.model.focal_length,
        principal_point=tuple(config.model.principal_point),
        image_size=tuple(config.model.image_size),
        triangulation_method=config.model.triangulation_method,
        min_triangulation_angle=config.model.min_triangulation_angle,
        max_reprojection_error=config.model.max_reprojection_error
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        max_epochs=config.train.max_epochs,
        optimizer=config.train.optimizer,
        scheduler=config.train.scheduler,
        save_every_n_epochs=config.train.save_every_n_epochs,
        save_top_k=config.train.save_top_k,
        monitor_metric=config.train.monitor_metric,
        monitor_mode=config.train.monitor_mode,
        patience=config.train.patience,
        min_delta=config.train.min_delta,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        reprojection_loss_weight=config.train.reprojection_loss_weight,
        triangulation_loss_weight=config.train.triangulation_loss_weight,
        regularization_loss_weight=config.train.regularization_loss_weight
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=get_device(config.device))
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if trainer.scheduler and checkpoint.get("scheduler_state_dict"):
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_metric = checkpoint["best_metric"]
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
