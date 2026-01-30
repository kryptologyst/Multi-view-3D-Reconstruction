# Multi-view 3D Reconstruction

A research-ready implementation of Structure from Motion (SfM) for multi-view 3D reconstruction using PyTorch, OpenCV, and advanced computer vision techniques.

## Overview

This project implements a comprehensive multi-view 3D reconstruction pipeline that can reconstruct 3D scenes from multiple 2D images taken from different viewpoints. The implementation includes both traditional SfM approaches and modern deep learning techniques, making it suitable for research, education, and practical applications.

## Features

- **Traditional SfM Pipeline**: Feature detection, matching, and triangulation using OpenCV
- **Synthetic Dataset Generation**: Create training data with known ground truth
- **Modern PyTorch Implementation**: GPU acceleration with automatic device detection
- **Comprehensive Evaluation**: Multiple metrics including Chamfer distance, PSNR, SSIM
- **Interactive Demo**: Streamlit-based web interface for easy experimentation
- **Production Ready**: Proper configuration management, logging, and checkpointing

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Apple Silicon MPS support (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Multi-view-3D-Reconstruction.git
cd Multi-view-3D-Reconstruction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Generate Synthetic Data

The project includes a synthetic dataset generator that creates multi-view scenes with known ground truth:

```python
from src.data.synthetic_dataset import SyntheticMultiViewDataset

# Create dataset
dataset = SyntheticMultiViewDataset(
    num_images=8,
    image_size=(640, 480),
    num_objects=5
)

# Get a sample
sample = dataset[0]
print(f"Generated {len(sample['points_3d'])} 3D points")
```

### 2. Train a Model

Train the SfM model on synthetic data:

```bash
python scripts/train.py --config configs/config.yaml
```

### 3. Evaluate Results

Evaluate the trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --results_dir results
```

### 4. Run Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   └── sfm_basic.py   # Basic SfM model
│   ├── data/              # Data loading and processing
│   │   └── synthetic_dataset.py
│   ├── utils/             # Utility functions
│   │   ├── device.py      # Device management
│   │   ├── geometry.py    # Geometric utilities
│   │   └── metrics.py     # Evaluation metrics
│   ├── train/             # Training utilities
│   │   └── trainer.py     # Training loop
│   └── eval/              # Evaluation utilities
│       └── evaluator.py   # Evaluation pipeline
├── configs/               # Configuration files
│   ├── config.yaml        # Main configuration
│   ├── model/             # Model configurations
│   ├── data/              # Data configurations
│   └── train/             # Training configurations
├── scripts/               # Training and evaluation scripts
├── demo/                  # Interactive demo
├── tests/                 # Unit tests
├── assets/                # Generated visualizations
└── data/                  # Data directory
    ├── raw/               # Raw data
    └── processed/         # Processed data
```

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/sfm_basic.yaml`: SfM model parameters
- `configs/data/synthetic.yaml`: Dataset parameters
- `configs/train/default.yaml`: Training parameters

### Key Parameters

**Model Parameters:**
- `feature_detector`: Feature detector type (sift, orb, surf)
- `matcher`: Feature matcher type (bf, flann)
- `min_matches`: Minimum number of matches required
- `ransac_threshold`: RANSAC threshold for robust estimation

**Data Parameters:**
- `num_images`: Number of images per scene
- `image_size`: Image dimensions
- `camera_radius`: Camera trajectory radius
- `num_objects`: Number of objects in synthetic scenes

**Training Parameters:**
- `learning_rate`: Learning rate
- `max_epochs`: Maximum training epochs
- `batch_size`: Batch size
- `mixed_precision`: Use mixed precision training

## Models

### SfMBasicModel

The basic Structure from Motion model implements:

1. **Feature Detection**: SIFT, ORB, or SURF feature detection
2. **Feature Matching**: Brute force or FLANN-based matching
3. **Fundamental Matrix Estimation**: RANSAC-based robust estimation
4. **Camera Pose Recovery**: Essential matrix decomposition
5. **Triangulation**: Direct Linear Transform (DLT) triangulation

**Key Features:**
- Automatic device detection (CUDA/MPS/CPU)
- Configurable feature detectors and matchers
- Robust RANSAC-based estimation
- Comprehensive error handling

## Evaluation Metrics

The project includes comprehensive evaluation metrics:

### 3D Metrics
- **Chamfer Distance**: Measures point cloud similarity
- **Earth Mover's Distance (EMD)**: Optimal transport distance
- **Point Cloud Accuracy**: Precision, recall, and F1 score
- **Completeness**: Fraction of ground truth points reconstructed

### Image Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Efficiency Metrics
- **FPS**: Frames per second during inference
- **Memory Usage**: Peak memory consumption
- **Model Size**: Number of parameters

## Dataset Schema

### Synthetic Dataset

The synthetic dataset generates multi-view scenes with:

**Input:**
- Multiple images from different viewpoints
- Camera poses and intrinsic parameters
- Ground truth 3D points

**Output:**
- Reconstructed 3D point cloud
- Estimated camera poses
- Evaluation metrics

**Data Format:**
```python
{
    "images": torch.Tensor,        # (N, H, W) - Input images
    "points_3d": torch.Tensor,     # (M, 3) - Ground truth 3D points
    "points_2d": torch.Tensor,     # (N, M, 2) - 2D projections
    "camera_poses": torch.Tensor,  # (N, 4, 4) - Camera poses
    "camera_matrix": torch.Tensor  # (3, 3) - Camera intrinsics
}
```

## Training Commands

### Basic Training
```bash
python scripts/train.py --config configs/config.yaml
```

### Resume Training
```bash
python scripts/train.py --config configs/config.yaml --resume checkpoints/latest.pt
```

### Custom Configuration
```bash
python scripts/train.py --config configs/config.yaml --device cuda --seed 123
```

## Evaluation Commands

### Basic Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt
```

### Custom Results Directory
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --results_dir my_results
```

## Demo Instructions

### Launch Streamlit Demo
```bash
streamlit run demo/app.py
```

### Demo Features
- Upload multiple images from different viewpoints
- Interactive 3D point cloud visualization
- Camera trajectory visualization
- Download reconstructed point clouds
- Real-time reconstruction statistics

### Demo Screenshots
The demo provides:
- Image upload interface
- 3D point cloud visualization with Plotly
- Camera trajectory overlay
- Reconstruction statistics and metrics
- Point cloud download functionality

## Performance and Efficiency

### Benchmarks
- **Inference Speed**: ~10-50 FPS depending on image size and device
- **Memory Usage**: ~2-8 GB VRAM for typical scenes
- **Model Size**: ~1-10 MB depending on configuration

### Device Support
- **CUDA**: Full GPU acceleration support
- **MPS**: Apple Silicon optimization
- **CPU**: Fallback CPU implementation

### Optimization Features
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large batches
- Automatic device detection and fallback
- Efficient data loading with multiple workers

## Known Limitations

1. **Feature Detection**: Limited to traditional feature detectors (SIFT, ORB, SURF)
2. **Scene Complexity**: Performance degrades with highly textured or repetitive scenes
3. **Camera Calibration**: Assumes known or estimated camera intrinsics
4. **Scale Ambiguity**: Reconstructed scenes have arbitrary scale
5. **Occlusion Handling**: Limited handling of occluded features

## Future Improvements

1. **Deep Learning Integration**: Add neural network-based feature detection
2. **NeRF Integration**: Incorporate Neural Radiance Fields for novel view synthesis
3. **Real-time Processing**: Optimize for real-time reconstruction
4. **Multi-scale Processing**: Handle scenes with large scale variations
5. **Uncertainty Quantification**: Add confidence estimates to reconstructions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multi_view_3d_reconstruction,
  title={Multi-view 3D Reconstruction: A Modern PyTorch Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Multi-view-3D-Reconstruction}
}
```

## Acknowledgments

- OpenCV community for computer vision algorithms
- PyTorch team for the deep learning framework
- Open3D team for 3D processing utilities
- Streamlit team for the web interface framework
# Multi-view-3D-Reconstruction
