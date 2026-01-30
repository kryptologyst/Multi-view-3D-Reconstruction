"""Evaluation module for 3D reconstruction models."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device
from ..utils.metrics import evaluate_reconstruction


class Evaluator:
    """Evaluator class for 3D reconstruction models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: str = "auto",
        save_results: bool = True,
        results_dir: str = "results"
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to use
            save_results: Whether to save results
            results_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = get_device(device)
        self.save_results = save_results
        self.results_dir = results_dir
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create results directory
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating model on device: {self.device}")
        
        all_metrics = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluation")):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch["images"])
                
                # Compute metrics for this batch
                if "points_3d" in batch and "points_3d" in outputs:
                    metrics = evaluate_reconstruction(
                        outputs["points_3d"].cpu().numpy(),
                        batch["points_3d"].cpu().numpy()
                    )
                    all_metrics.append(metrics)
                
                # Store predictions for analysis
                prediction = {
                    "idx": batch["idx"].cpu().numpy(),
                    "points_3d_pred": outputs["points_3d"].cpu().numpy(),
                    "camera_poses": outputs["camera_poses"].cpu().numpy(),
                    "camera_matrix": outputs["camera_matrix"].cpu().numpy()
                }
                
                if "points_3d" in batch:
                    prediction["points_3d_gt"] = batch["points_3d"].cpu().numpy()
                
                all_predictions.append(prediction)
        
        # Aggregate metrics
        if all_metrics:
            aggregated_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated_metrics[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        else:
            aggregated_metrics = {}
        
        # Save results
        if self.save_results:
            self._save_results(aggregated_metrics, all_predictions)
        
        return aggregated_metrics
    
    def _save_results(self, metrics: Dict[str, Dict[str, float]], predictions: List[Dict]):
        """Save evaluation results.
        
        Args:
            metrics: Aggregated metrics
            predictions: Model predictions
        """
        # Save metrics
        metrics_path = os.path.join(self.results_dir, "metrics.npz")
        np.savez(metrics_path, **metrics)
        
        # Save predictions
        predictions_path = os.path.join(self.results_dir, "predictions.npz")
        np.savez(predictions_path, predictions=predictions)
        
        # Save summary
        summary_path = os.path.join(self.results_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Evaluation Summary\n")
            f.write("================\n\n")
            
            for metric_name, metric_stats in metrics.items():
                f.write(f"{metric_name}:\n")
                f.write(f"  Mean: {metric_stats['mean']:.6f}\n")
                f.write(f"  Std:  {metric_stats['std']:.6f}\n")
                f.write(f"  Min:  {metric_stats['min']:.6f}\n")
                f.write(f"  Max:  {metric_stats['max']:.6f}\n\n")
        
        print(f"Results saved to {self.results_dir}")
    
    def compute_efficiency_metrics(self) -> Dict[str, float]:
        """Compute efficiency metrics (FPS, memory usage, etc.).
        
        Returns:
            dict: Efficiency metrics
        """
        import time
        import psutil
        
        # Warm up
        dummy_batch = next(iter(self.test_loader))
        dummy_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in dummy_batch.items()}
        
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(dummy_batch["images"])
        
        # Measure inference time
        times = []
        memory_usage = []
        
        for batch in self.test_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            # Measure time
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(batch["images"])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Measure memory after
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append(memory_after - memory_before)
            
            times.append(end_time - start_time)
        
        # Compute metrics
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        efficiency_metrics = {
            "fps": fps,
            "avg_inference_time": avg_time,
            "std_inference_time": np.std(times)
        }
        
        if memory_usage:
            efficiency_metrics["avg_memory_usage"] = np.mean(memory_usage)
            efficiency_metrics["max_memory_usage"] = np.max(memory_usage)
        
        # Model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        efficiency_metrics["total_parameters"] = total_params
        efficiency_metrics["trainable_parameters"] = trainable_params
        
        return efficiency_metrics
