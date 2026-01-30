"""Synthetic multi-view dataset for 3D reconstruction."""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.geometry import create_camera_matrix, create_camera_poses, project_points


class SyntheticMultiViewDataset(Dataset):
    """Synthetic multi-view dataset for 3D reconstruction.
    
    This dataset generates synthetic scenes with multiple camera viewpoints
    and corresponding 3D point clouds for training and evaluation.
    """
    
    def __init__(
        self,
        num_images: int = 8,
        image_size: Tuple[int, int] = (640, 480),
        noise_level: float = 0.01,
        blur_sigma: float = 0.5,
        camera_radius: float = 2.0,
        camera_height: float = 1.0,
        look_at_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        num_objects: int = 5,
        object_scale_range: Tuple[float, float] = (0.5, 2.0),
        object_position_range: Tuple[float, float] = (-1.0, 1.0),
        focal_length: float = 1000.0,
        principal_point: Tuple[float, float] = (320.0, 240.0),
        split: str = "train",
        num_samples: int = 1000
    ):
        """Initialize synthetic dataset.
        
        Args:
            num_images: Number of images per scene
            image_size: Image size (width, height)
            noise_level: Gaussian noise level
            blur_sigma: Gaussian blur sigma
            camera_radius: Radius of camera trajectory
            camera_height: Height of camera trajectory
            look_at_center: Center point for camera look-at
            num_objects: Number of objects in the scene
            object_scale_range: Range of object scales
            object_position_range: Range of object positions
            focal_length: Camera focal length
            principal_point: Camera principal point
            split: Dataset split ('train', 'val', 'test')
            num_samples: Number of samples in the dataset
        """
        self.num_images = num_images
        self.image_size = image_size
        self.noise_level = noise_level
        self.blur_sigma = blur_sigma
        self.camera_radius = camera_radius
        self.camera_height = camera_height
        self.look_at_center = look_at_center
        self.num_objects = num_objects
        self.object_scale_range = object_scale_range
        self.object_position_range = object_position_range
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.split = split
        self.num_samples = num_samples
        
        # Create camera intrinsic matrix
        self.camera_matrix = create_camera_matrix(
            focal_length, principal_point, image_size
        )
        
        # Generate camera poses
        self.camera_poses = create_camera_poses(
            num_images, camera_radius, camera_height, look_at_center
        )
    
    def generate_scene(self) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Generate a synthetic scene with 3D points and multi-view images.
        
        Returns:
            Tuple containing:
                - points_3d: 3D points (N x 3)
                - images: List of images (num_images x H x W)
                - points_2d: List of 2D projections (num_images x N x 2)
        """
        # Generate 3D points (simple geometric shapes)
        points_3d = self._generate_3d_points()
        
        # Project points to each camera view
        images = []
        points_2d_list = []
        
        for pose in self.camera_poses:
            # Project 3D points to 2D
            points_2d = project_points(points_3d, self.camera_matrix, pose)
            
            # Create image with projected points
            image = self._create_image_from_points(points_2d)
            
            # Add noise and blur
            image = self._add_noise_and_blur(image)
            
            images.append(image)
            points_2d_list.append(points_2d)
        
        return points_3d, images, points_2d_list
    
    def _generate_3d_points(self) -> np.ndarray:
        """Generate 3D points for the scene.
        
        Returns:
            np.ndarray: 3D points (N x 3)
        """
        points_3d = []
        
        # Generate points for each object
        for _ in range(self.num_objects):
            # Random object center
            center = np.random.uniform(
                self.object_position_range[0],
                self.object_position_range[1],
                3
            )
            
            # Random scale
            scale = np.random.uniform(
                self.object_scale_range[0],
                self.object_scale_range[1]
            )
            
            # Generate points for a simple shape (cube)
            cube_points = self._generate_cube_points(center, scale)
            points_3d.extend(cube_points)
        
        return np.array(points_3d)
    
    def _generate_cube_points(
        self,
        center: np.ndarray,
        scale: float,
        num_points: int = 100
    ) -> List[np.ndarray]:
        """Generate points for a cube.
        
        Args:
            center: Cube center (3,)
            scale: Cube scale
            num_points: Number of points per cube
            
        Returns:
            List[np.ndarray]: List of 3D points
        """
        points = []
        
        for _ in range(num_points):
            # Random point on cube surface
            face = np.random.randint(0, 6)
            
            if face == 0:  # Front face
                x = np.random.uniform(-scale, scale)
                y = np.random.uniform(-scale, scale)
                z = scale
            elif face == 1:  # Back face
                x = np.random.uniform(-scale, scale)
                y = np.random.uniform(-scale, scale)
                z = -scale
            elif face == 2:  # Left face
                x = -scale
                y = np.random.uniform(-scale, scale)
                z = np.random.uniform(-scale, scale)
            elif face == 3:  # Right face
                x = scale
                y = np.random.uniform(-scale, scale)
                z = np.random.uniform(-scale, scale)
            elif face == 4:  # Top face
                x = np.random.uniform(-scale, scale)
                y = scale
                z = np.random.uniform(-scale, scale)
            else:  # Bottom face
                x = np.random.uniform(-scale, scale)
                y = -scale
                z = np.random.uniform(-scale, scale)
            
            point = np.array([x, y, z]) + center
            points.append(point)
        
        return points
    
    def _create_image_from_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Create image from 2D points.
        
        Args:
            points_2d: 2D points (N x 2)
            
        Returns:
            np.ndarray: Image (H x W)
        """
        image = np.zeros(self.image_size[::-1], dtype=np.float32)
        
        for point in points_2d:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                # Draw a small circle around the point
                cv2.circle(image, (x, y), 2, 1.0, -1)
        
        return image
    
    def _add_noise_and_blur(self, image: np.ndarray) -> np.ndarray:
        """Add noise and blur to image.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Processed image
        """
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, image.shape)
        image = image + noise
        
        # Add Gaussian blur
        if self.blur_sigma > 0:
            image = cv2.GaussianBlur(image, (5, 5), self.blur_sigma)
        
        # Clip values
        image = np.clip(image, 0, 1)
        
        return image
    
    def __len__(self) -> int:
        """Return dataset length."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Sample containing images, points_3d, points_2d, camera_poses
        """
        # Set random seed for reproducibility
        np.random.seed(idx)
        
        # Generate scene
        points_3d, images, points_2d_list = self.generate_scene()
        
        # Convert to tensors
        sample = {
            "images": torch.stack([torch.from_numpy(img).float() for img in images]),
            "points_3d": torch.from_numpy(points_3d).float(),
            "points_2d": torch.stack([torch.from_numpy(pts).float() for pts in points_2d_list]),
            "camera_poses": torch.stack([torch.from_numpy(pose).float() for pose in self.camera_poses]),
            "camera_matrix": torch.from_numpy(self.camera_matrix).float(),
            "idx": torch.tensor(idx)
        }
        
        return sample
