"""Structure from Motion (SfM) model for 3D reconstruction."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..utils.geometry import triangulate_points, compute_reprojection_error


class SfMBasicModel(nn.Module):
    """Basic Structure from Motion model for 3D reconstruction.
    
    This model implements a traditional SfM pipeline using feature detection,
    matching, and triangulation for multi-view 3D reconstruction.
    """
    
    def __init__(
        self,
        feature_detector: str = "sift",
        matcher: str = "bf",
        min_matches: int = 50,
        ransac_threshold: float = 1.0,
        confidence: float = 0.99,
        focal_length: float = 1000.0,
        principal_point: Tuple[float, float] = (320.0, 240.0),
        image_size: Tuple[int, int] = (640, 480),
        triangulation_method: str = "direct_linear_transform",
        min_triangulation_angle: float = 1.0,
        max_reprojection_error: float = 1.0
    ):
        """Initialize SfM model.
        
        Args:
            feature_detector: Feature detector type ('sift', 'orb', 'surf')
            matcher: Matcher type ('bf', 'flann')
            min_matches: Minimum number of matches required
            ransac_threshold: RANSAC threshold for fundamental matrix
            confidence: Confidence level for RANSAC
            focal_length: Camera focal length
            principal_point: Camera principal point
            image_size: Image size (width, height)
            triangulation_method: Triangulation method
            min_triangulation_angle: Minimum triangulation angle in degrees
            max_reprojection_error: Maximum reprojection error
        """
        super().__init__()
        
        self.feature_detector = feature_detector
        self.matcher = matcher
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.triangulation_method = triangulation_method
        self.min_triangulation_angle = min_triangulation_angle
        self.max_reprojection_error = max_reprojection_error
        
        # Initialize feature detector
        self._init_feature_detector()
        
        # Initialize matcher
        self._init_matcher()
        
        # Camera intrinsic matrix
        self.camera_matrix = self._create_camera_matrix()
    
    def _init_feature_detector(self):
        """Initialize feature detector."""
        if self.feature_detector == "sift":
            self.detector = cv2.SIFT_create()
        elif self.feature_detector == "orb":
            self.detector = cv2.ORB_create()
        elif self.feature_detector == "surf":
            self.detector = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"Unknown feature detector: {self.feature_detector}")
    
    def _init_matcher(self):
        """Initialize feature matcher."""
        if self.matcher == "bf":
            if self.feature_detector == "orb":
                self.matcher_obj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher_obj = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif self.matcher == "flann":
            if self.feature_detector == "orb":
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                 table_number=6,
                                 key_size=12,
                                 multi_probe_level=1)
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            
            search_params = dict(checks=50)
            self.matcher_obj = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher: {self.matcher}")
    
    def _create_camera_matrix(self) -> np.ndarray:
        """Create camera intrinsic matrix."""
        fx = fy = self.focal_length
        cx, cy = self.principal_point
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def detect_and_compute_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect and compute features for an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple containing keypoints and descriptors
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:
        """Match features between two images.
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            
        Returns:
            List of matches
        """
        if self.matcher == "bf":
            matches = self.matcher_obj.match(descriptors1, descriptors2)
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
        else:
            matches = self.matcher_obj.knnMatch(descriptors1, descriptors2, k=2)
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            matches = good_matches
        
        return matches
    
    def estimate_fundamental_matrix(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate fundamental matrix between two views.
        
        Args:
            points1: Points in first image (N x 2)
            points2: Points in second image (N x 2)
            
        Returns:
            Tuple containing fundamental matrix and mask
        """
        F, mask = cv2.findFundamentalMat(
            points1, points2,
            cv2.FM_RANSAC,
            self.ransac_threshold,
            self.confidence
        )
        
        return F, mask
    
    def estimate_essential_matrix(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate essential matrix between two views.
        
        Args:
            points1: Points in first image (N x 2)
            points2: Points in second image (N x 2)
            
        Returns:
            Tuple containing essential matrix and mask
        """
        E, mask = cv2.findEssentialMat(
            points1, points2,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=self.confidence,
            threshold=self.ransac_threshold
        )
        
        return E, mask
    
    def recover_pose(
        self,
        E: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recover camera pose from essential matrix.
        
        Args:
            E: Essential matrix
            points1: Points in first image (N x 2)
            points2: Points in second image (N x 2)
            
        Returns:
            Tuple containing rotation, translation, and mask
        """
        points, R, t, mask = cv2.recoverPose(
            E, points1, points2, self.camera_matrix
        )
        
        return R, t, mask
    
    def triangulate_points(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """Triangulate 3D points from two views.
        
        Args:
            P1: Camera projection matrix for view 1 (3x4)
            P2: Camera projection matrix for view 2 (3x4)
            points1: Image points in view 1 (Nx2)
            points2: Image points in view 2 (Nx2)
            
        Returns:
            np.ndarray: Triangulated 3D points (Nx3)
        """
        if self.triangulation_method == "direct_linear_transform":
            return triangulate_points(P1, P2, points1, points2)
        else:
            # Use OpenCV triangulation
            points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
            points_3d = points_4d[:3] / points_4d[3]
            return points_3d.T
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for SfM reconstruction.
        
        Args:
            images: Input images (B x N x H x W)
            
        Returns:
            dict: Reconstruction results
        """
        batch_size, num_images, height, width = images.shape
        batch_results = []
        
        for b in range(batch_size):
            # Convert to numpy for OpenCV processing
            batch_images = images[b].cpu().numpy()
            
            # Process each image pair
            all_points_3d = []
            all_camera_poses = []
            
            # Use first image as reference
            ref_image = (batch_images[0] * 255).astype(np.uint8)
            ref_keypoints, ref_descriptors = self.detect_and_compute_features(ref_image)
            
            if ref_descriptors is None or len(ref_descriptors) < self.min_matches:
                continue
            
            # Initialize with identity pose for reference camera
            ref_pose = np.eye(4)
            all_camera_poses.append(ref_pose)
            
            for i in range(1, num_images):
                curr_image = (batch_images[i] * 255).astype(np.uint8)
                curr_keypoints, curr_descriptors = self.detect_and_compute_features(curr_image)
                
                if curr_descriptors is None or len(curr_descriptors) < self.min_matches:
                    continue
                
                # Match features
                matches = self.match_features(ref_descriptors, curr_descriptors)
                
                if len(matches) < self.min_matches:
                    continue
                
                # Extract matched points
                ref_points = np.float32([ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                
                # Estimate essential matrix
                E, mask = self.estimate_essential_matrix(ref_points, curr_points)
                
                if E is None:
                    continue
                
                # Recover pose
                R, t, pose_mask = self.recover_pose(E, ref_points, curr_points)
                
                if R is None or t is None:
                    continue
                
                # Create pose matrix
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t.flatten()
                all_camera_poses.append(pose)
                
                # Triangulate points
                P1 = self.camera_matrix @ ref_pose[:3]
                P2 = self.camera_matrix @ pose[:3]
                
                # Filter points based on mask
                if pose_mask is not None:
                    ref_points = ref_points[pose_mask.ravel() == 1]
                    curr_points = curr_points[pose_mask.ravel() == 1]
                
                if len(ref_points) > 0:
                    points_3d = self.triangulate_points(P1, P2, ref_points, curr_points)
                    all_points_3d.extend(points_3d)
            
            # Convert results to tensors
            if all_points_3d:
                points_3d_tensor = torch.from_numpy(np.array(all_points_3d)).float()
                poses_tensor = torch.from_numpy(np.array(all_camera_poses)).float()
            else:
                points_3d_tensor = torch.zeros(0, 3).float()
                poses_tensor = torch.zeros(0, 4, 4).float()
            
            batch_results.append({
                "points_3d": points_3d_tensor,
                "camera_poses": poses_tensor,
                "camera_matrix": torch.from_numpy(self.camera_matrix).float()
            })
        
        return batch_results[0] if batch_results else {
            "points_3d": torch.zeros(0, 3).float(),
            "camera_poses": torch.zeros(0, 4, 4).float(),
            "camera_matrix": torch.from_numpy(self.camera_matrix).float()
        }
