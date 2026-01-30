"""Streamlit demo for multi-view 3D reconstruction."""

import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sfm_basic import SfMBasicModel
from src.utils.device import get_device, get_device_info
from src.utils.metrics import evaluate_reconstruction


def load_model() -> SfMBasicModel:
    """Load the SfM model."""
    model = SfMBasicModel()
    device = get_device("auto")
    model.to(device)
    model.eval()
    return model


def process_images(images: List[np.ndarray], model: SfMBasicModel) -> dict:
    """Process images through the model.
    
    Args:
        images: List of input images
        model: SfM model
        
    Returns:
        dict: Reconstruction results
    """
    # Convert images to tensor format
    images_tensor = torch.stack([
        torch.from_numpy(img).float() / 255.0 
        for img in images
    ]).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    device = get_device("auto")
    images_tensor = images_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        results = model(images_tensor)
    
    return results


def visualize_point_cloud(points_3d: np.ndarray) -> str:
    """Create a visualization of the point cloud.
    
    Args:
        points_3d: 3D points (N x 3)
        
    Returns:
        str: Path to saved visualization
    """
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Color points by height
        colors = np.zeros_like(points_3d)
        colors[:, 2] = (points_3d[:, 2] - points_3d[:, 2].min()) / (points_3d[:, 2].max() - points_3d[:, 2].min())
        colors[:, 1] = 1.0 - colors[:, 2]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save visualization
        temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        o3d.io.write_point_cloud(temp_file.name, pcd)
        
        return temp_file.name
        
    except ImportError:
        st.warning("Open3D not available. Install with: pip install open3d")
        return None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-view 3D Reconstruction",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Multi-view 3D Reconstruction")
    st.markdown("Upload multiple images from different viewpoints to reconstruct a 3D scene.")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    feature_detector = st.sidebar.selectbox(
        "Feature Detector",
        ["sift", "orb"],
        index=0
    )
    min_matches = st.sidebar.slider(
        "Minimum Matches",
        min_value=10,
        max_value=200,
        value=50
    )
    ransac_threshold = st.sidebar.slider(
        "RANSAC Threshold",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    
    # Device info
    device_info = get_device_info()
    st.sidebar.subheader("Device Information")
    st.sidebar.write(f"CUDA Available: {device_info['cuda_available']}")
    st.sidebar.write(f"MPS Available: {device_info['mps_available']}")
    if device_info['cuda_available']:
        st.sidebar.write(f"CUDA Device: {device_info.get('cuda_device', 'Unknown')}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Images")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload 2 or more images from different viewpoints"
        )
        
        if uploaded_files:
            # Display uploaded images
            st.subheader("Uploaded Images")
            images = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Read image
                image = Image.open(uploaded_file)
                image_np = np.array(image.convert('RGB'))
                images.append(image_np)
                
                # Display image
                st.image(image, caption=f"Image {i+1}", use_column_width=True)
            
            # Process images
            if len(images) >= 2:
                if st.button("Reconstruct 3D Scene", type="primary"):
                    with st.spinner("Processing images..."):
                        # Load model
                        model = load_model()
                        
                        # Process images
                        results = process_images(images, model)
                        
                        # Extract results
                        points_3d = results["points_3d"].cpu().numpy()
                        camera_poses = results["camera_poses"].cpu().numpy()
                        
                        # Store results in session state
                        st.session_state['points_3d'] = points_3d
                        st.session_state['camera_poses'] = camera_poses
                        st.session_state['images'] = images
                        
                        st.success(f"Reconstruction complete! Found {len(points_3d)} 3D points.")
            else:
                st.warning("Please upload at least 2 images for reconstruction.")
    
    with col2:
        st.header("3D Reconstruction Results")
        
        if 'points_3d' in st.session_state:
            points_3d = st.session_state['points_3d']
            camera_poses = st.session_state['camera_poses']
            images = st.session_state['images']
            
            # Display statistics
            st.subheader("Reconstruction Statistics")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric("3D Points", len(points_3d))
                st.metric("Camera Poses", len(camera_poses))
            
            with col2_2:
                if len(points_3d) > 0:
                    st.metric("X Range", f"{points_3d[:, 0].min():.2f} to {points_3d[:, 0].max():.2f}")
                    st.metric("Y Range", f"{points_3d[:, 1].min():.2f} to {points_3d[:, 1].max():.2f}")
                    st.metric("Z Range", f"{points_3d[:, 2].min():.2f} to {points_3d[:, 2].max():.2f}")
            
            # Point cloud visualization
            st.subheader("Point Cloud Visualization")
            
            if len(points_3d) > 0:
                # Create visualization
                viz_path = visualize_point_cloud(points_3d)
                
                if viz_path:
                    # Download button for point cloud
                    with open(viz_path, 'rb') as f:
                        st.download_button(
                            label="Download Point Cloud (.ply)",
                            data=f.read(),
                            file_name="reconstructed_point_cloud.ply",
                            mime="application/octet-stream"
                        )
                    
                    # Clean up
                    os.unlink(viz_path)
                
                # Simple 3D scatter plot using plotly
                try:
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=[go.Scatter3d(
                        x=points_3d[:, 0],
                        y=points_3d[:, 1],
                        z=points_3d[:, 2],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=points_3d[:, 2],
                            colorscale='Viridis',
                            opacity=0.8
                        )
                    )])
                    
                    fig.update_layout(
                        title="3D Point Cloud",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Z"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except ImportError:
                    st.warning("Plotly not available. Install with: pip install plotly")
            
            # Camera trajectory visualization
            if len(camera_poses) > 1:
                st.subheader("Camera Trajectory")
                
                try:
                    import plotly.graph_objects as go
                    
                    # Extract camera positions
                    camera_positions = camera_poses[:, :3, 3]
                    
                    fig = go.Figure(data=[go.Scatter3d(
                        x=camera_positions[:, 0],
                        y=camera_positions[:, 1],
                        z=camera_positions[:, 2],
                        mode='markers+lines',
                        marker=dict(size=8, color='red'),
                        line=dict(color='red', width=2),
                        name='Camera Trajectory'
                    )])
                    
                    # Add point cloud
                    if len(points_3d) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=points_3d[:, 0],
                            y=points_3d[:, 1],
                            z=points_3d[:, 2],
                            mode='markers',
                            marker=dict(size=1, color='blue', opacity=0.5),
                            name='3D Points'
                        ))
                    
                    fig.update_layout(
                        title="Camera Trajectory and 3D Points",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Z"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except ImportError:
                    st.warning("Plotly not available. Install with: pip install plotly")
            
            # Evaluation metrics (if ground truth available)
            st.subheader("Evaluation Metrics")
            st.info("Evaluation metrics would be computed if ground truth 3D points were available.")
            
        else:
            st.info("Upload images and run reconstruction to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Multi-view 3D Reconstruction Demo** - Built with PyTorch, OpenCV, and Streamlit")


if __name__ == "__main__":
    main()
