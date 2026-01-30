#!/usr/bin/env python3
"""Evaluation script for multi-view 3D reconstruction."""

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
from src.eval.evaluator import Evaluator
from src.utils.device import get_device, set_seed


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate multi-view 3D reconstruction model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    
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
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = SyntheticMultiViewDataset(
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
        split="test",
        num_samples=500
    )
    
    # Create test data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one sample at a time for evaluation
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
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=get_device(config.device))
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=config.device,
        save_results=True,
        results_dir=args.results_dir
    )
    
    # Run evaluation
    print("Running evaluation...")
    metrics = evaluator.evaluate()
    
    # Print results
    print("\nEvaluation Results:")
    print("==================")
    for metric_name, metric_stats in metrics.items():
        print(f"{metric_name}:")
        print(f"  Mean: {metric_stats['mean']:.6f}")
        print(f"  Std:  {metric_stats['std']:.6f}")
        print(f"  Min:  {metric_stats['min']:.6f}")
        print(f"  Max:  {metric_stats['max']:.6f}")
        print()
    
    # Compute efficiency metrics
    print("Computing efficiency metrics...")
    efficiency_metrics = evaluator.compute_efficiency_metrics()
    
    print("\nEfficiency Metrics:")
    print("==================")
    for metric_name, value in efficiency_metrics.items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    
    print(f"\nResults saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
