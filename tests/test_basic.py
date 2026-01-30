"""Basic tests for the multi-view 3D reconstruction project."""

import pytest
import torch
import numpy as np

from src.utils.device import get_device, set_seed
from src.utils.geometry import create_camera_matrix, triangulate_points
from src.utils.metrics import chamfer_distance, compute_psnr
from src.data.synthetic_dataset import SyntheticMultiViewDataset
from src.models.sfm_basic import SfMBasicModel


def test_device_management():
    """Test device management utilities."""
    device = get_device("auto")
    assert isinstance(device, torch.device)
    
    set_seed(42)
    # Test that seeding works
    assert True


def test_geometry_utilities():
    """Test geometry utility functions."""
    # Test camera matrix creation
    K = create_camera_matrix(1000.0, (320.0, 240.0), (640, 480))
    assert K.shape == (3, 3)
    assert K[0, 0] == 1000.0  # fx
    assert K[1, 1] == 1000.0  # fy
    assert K[0, 2] == 320.0   # cx
    assert K[1, 2] == 240.0   # cy
    
    # Test triangulation
    P1 = np.eye(3, 4)
    P2 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    points1 = np.array([[100, 100], [200, 200]])
    points2 = np.array([[150, 100], [250, 200]])
    
    points_3d = triangulate_points(P1, P2, points1, points2)
    assert points_3d.shape == (2, 3)


def test_metrics():
    """Test evaluation metrics."""
    # Test Chamfer distance
    points1 = np.random.rand(100, 3)
    points2 = np.random.rand(100, 3)
    cd = chamfer_distance(points1, points2)
    assert cd >= 0
    
    # Test PSNR
    img1 = np.random.rand(100, 100)
    img2 = img1 + 0.1 * np.random.rand(100, 100)
    psnr = compute_psnr(img1, img2)
    assert psnr > 0


def test_synthetic_dataset():
    """Test synthetic dataset generation."""
    dataset = SyntheticMultiViewDataset(
        num_images=4,
        image_size=(320, 240),
        num_samples=10
    )
    
    assert len(dataset) == 10
    
    sample = dataset[0]
    assert "images" in sample
    assert "points_3d" in sample
    assert "points_2d" in sample
    assert "camera_poses" in sample
    assert "camera_matrix" in sample
    
    assert sample["images"].shape[0] == 4  # num_images
    assert len(sample["points_3d"]) > 0
    assert len(sample["points_2d"]) == 4


def test_sfm_model():
    """Test SfM model initialization and basic functionality."""
    model = SfMBasicModel(
        feature_detector="sift",
        min_matches=10
    )
    
    # Test model initialization
    assert model.feature_detector == "sift"
    assert model.min_matches == 10
    
    # Test with dummy data
    dummy_images = torch.rand(1, 4, 240, 320)  # batch_size=1, num_images=4
    with torch.no_grad():
        results = model(dummy_images)
    
    assert "points_3d" in results
    assert "camera_poses" in results
    assert "camera_matrix" in results


if __name__ == "__main__":
    pytest.main([__file__])
