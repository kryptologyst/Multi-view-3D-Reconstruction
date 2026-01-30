"""Evaluation metrics for 3D reconstruction."""

from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist


def chamfer_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    bidirectional: bool = True
) -> float:
    """Compute Chamfer distance between two point clouds.
    
    Args:
        points1: First point cloud (N1 x 3)
        points2: Second point cloud (N2 x 3)
        bidirectional: Whether to compute bidirectional distance
        
    Returns:
        float: Chamfer distance
    """
    # Compute pairwise distances
    dist_matrix = cdist(points1, points2)
    
    # Distance from points1 to points2
    dist1_to_2 = np.min(dist_matrix, axis=1)
    chamfer_1_to_2 = np.mean(dist1_to_2)
    
    if not bidirectional:
        return chamfer_1_to_2
    
    # Distance from points2 to points1
    dist2_to_1 = np.min(dist_matrix, axis=0)
    chamfer_2_to_1 = np.mean(dist2_to_1)
    
    return chamfer_1_to_2 + chamfer_2_to_1


def earth_mover_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    max_iter: int = 100
) -> float:
    """Compute Earth Mover's Distance (EMD) between two point clouds.
    
    Args:
        points1: First point cloud (N1 x 3)
        points2: Second point cloud (N2 x 3)
        max_iter: Maximum number of iterations
        
    Returns:
        float: EMD value
    """
    from scipy.optimize import linear_sum_assignment
    
    # Compute pairwise distances
    dist_matrix = cdist(points1, points2)
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(dist_matrix)
    
    # Compute EMD
    emd = np.sum(dist_matrix[row_indices, col_indices]) / len(points1)
    
    return emd


def point_cloud_accuracy(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    threshold: float = 0.01
) -> Dict[str, float]:
    """Compute point cloud accuracy metrics.
    
    Args:
        points_pred: Predicted point cloud (N x 3)
        points_gt: Ground truth point cloud (M x 3)
        threshold: Distance threshold for accuracy
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Compute pairwise distances
    dist_matrix = cdist(points_pred, points_gt)
    
    # Accuracy: fraction of predicted points within threshold
    min_distances = np.min(dist_matrix, axis=1)
    accuracy = np.mean(min_distances < threshold)
    
    # Completeness: fraction of GT points within threshold
    min_distances_gt = np.min(dist_matrix, axis=0)
    completeness = np.mean(min_distances_gt < threshold)
    
    # F1 score
    f1 = 2 * accuracy * completeness / (accuracy + completeness + 1e-8)
    
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "f1_score": f1
    }


def compute_psnr(
    image1: np.ndarray,
    image2: np.ndarray,
    max_value: float = 1.0
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        image1: First image
        image2: Second image
        max_value: Maximum possible pixel value
        
    Returns:
        float: PSNR value in dB
    """
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
    return psnr


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    """Compute Structural Similarity Index (SSIM).
    
    Args:
        image1: First image
        image2: Second image
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
        
    Returns:
        float: SSIM value
    """
    from skimage.metrics import structural_similarity
    
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        image1 = np.mean(image1, axis=2)
    if len(image2.shape) == 3:
        image2 = np.mean(image2, axis=2)
    
    ssim = structural_similarity(
        image1, image2,
        win_size=window_size,
        sigma=sigma,
        data_range=image1.max() - image1.min()
    )
    
    return ssim


def evaluate_reconstruction(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    images_pred: Optional[List[np.ndarray]] = None,
    images_gt: Optional[List[np.ndarray]] = None
) -> Dict[str, float]:
    """Comprehensive evaluation of 3D reconstruction.
    
    Args:
        points_pred: Predicted 3D points (N x 3)
        points_gt: Ground truth 3D points (M x 3)
        images_pred: Predicted rendered images (optional)
        images_gt: Ground truth images (optional)
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    metrics = {}
    
    # 3D metrics
    metrics["chamfer_distance"] = chamfer_distance(points_pred, points_gt)
    metrics["earth_mover_distance"] = earth_mover_distance(points_pred, points_gt)
    
    # Point cloud accuracy
    pca_metrics = point_cloud_accuracy(points_pred, points_gt)
    metrics.update(pca_metrics)
    
    # Image metrics (if provided)
    if images_pred is not None and images_gt is not None:
        psnr_values = []
        ssim_values = []
        
        for img_pred, img_gt in zip(images_pred, images_gt):
            psnr_values.append(compute_psnr(img_pred, img_gt))
            ssim_values.append(compute_ssim(img_pred, img_gt))
        
        metrics["psnr_mean"] = np.mean(psnr_values)
        metrics["ssim_mean"] = np.mean(ssim_values)
    
    return metrics
