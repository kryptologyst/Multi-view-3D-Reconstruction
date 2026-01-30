"""Camera and geometry utilities for 3D reconstruction."""

from typing import List, Tuple, Union

import numpy as np
import torch


def create_camera_matrix(
    focal_length: Union[float, Tuple[float, float]],
    principal_point: Tuple[float, float],
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Create camera intrinsic matrix.
    
    Args:
        focal_length: Focal length(s) in pixels
        principal_point: Principal point (cx, cy) in pixels
        image_size: Image size (width, height)
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix
    """
    if isinstance(focal_length, (int, float)):
        fx = fy = focal_length
    else:
        fx, fy = focal_length
    
    cx, cy = principal_point
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def create_camera_poses(
    num_poses: int,
    radius: float = 2.0,
    height: float = 1.0,
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[np.ndarray]:
    """Create camera poses in a circular trajectory.
    
    Args:
        num_poses: Number of camera poses
        radius: Radius of the circular trajectory
        height: Height of the camera trajectory
        look_at: Point to look at (x, y, z)
        
    Returns:
        List[np.ndarray]: List of 4x4 camera pose matrices
    """
    poses = []
    look_at = np.array(look_at)
    
    for i in range(num_poses):
        angle = 2 * np.pi * i / num_poses
        
        # Camera position in circular trajectory
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height
        
        camera_pos = np.array([x, y, z])
        
        # Create look-at matrix
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create rotation matrix
        R = np.column_stack([right, up, -forward])
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = camera_pos
        
        poses.append(pose)
    
    return poses


def triangulate_points(
    P1: np.ndarray,
    P2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray
) -> np.ndarray:
    """Triangulate 3D points from two views using DLT.
    
    Args:
        P1: Camera projection matrix for view 1 (3x4)
        P2: Camera projection matrix for view 2 (3x4)
        points1: Image points in view 1 (Nx2)
        points2: Image points in view 2 (Nx2)
        
    Returns:
        np.ndarray: Triangulated 3D points (Nx3)
    """
    points_3d = []
    
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # Build system of equations Ax = 0
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])
        
        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        point_3d = V[-1, :3] / V[-1, 3]
        points_3d.append(point_3d)
    
    return np.array(points_3d)


def project_points(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    pose: np.ndarray
) -> np.ndarray:
    """Project 3D points to image coordinates.
    
    Args:
        points_3d: 3D points (Nx3)
        camera_matrix: Camera intrinsic matrix (3x3)
        pose: Camera pose matrix (4x4)
        
    Returns:
        np.ndarray: Projected 2D points (Nx2)
    """
    # Transform points to camera coordinates
    points_3d_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    points_cam = (pose @ points_3d_homo.T).T[:, :3]
    
    # Project to image plane
    points_2d_homo = (camera_matrix @ points_cam.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    return points_2d


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    pose: np.ndarray
) -> float:
    """Compute reprojection error for 3D points.
    
    Args:
        points_3d: 3D points (Nx3)
        points_2d: Observed 2D points (Nx2)
        camera_matrix: Camera intrinsic matrix (3x3)
        pose: Camera pose matrix (4x4)
        
    Returns:
        float: Mean reprojection error in pixels
    """
    projected_points = project_points(points_3d, camera_matrix, pose)
    errors = np.linalg.norm(projected_points - points_2d, axis=1)
    return np.mean(errors)
