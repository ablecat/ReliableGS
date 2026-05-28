# ReliableGS: Multi-View Geometric Reliability for Transparent Object Gaussian Splatting

This repository contains the code for ReliableGS, a reliability-guided framework for transparent object reconstruction built upon 3D Gaussian Splatting.

## Environment Setup

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (tested with CUDA 12.1)
- PyTorch 2.0+ with CUDA support

### Installation

```bash
# Create conda environment
conda create -n reliablegs python=3.10 -y
conda activate reliablegs

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Build CUDA submodules
pip install submodules/diff-first-surface-rasterization
pip install submodules/simple-knn
```

### Additional Dependencies
- **StableNormal**: Used for single-view normal priors. Follow [StableNormal](https://github.com/Stable-X/StableNormal) to set up and generate normal maps.
- **Grounded-SAM2**: Used for transparent region mask generation. Follow [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) to generate masks.

## Dataset Preparation

### TransLab Dataset
1. Download TransLab from [TSGS](https://github.com/Lmw-TSGS/TSGS).
2. Organize as follows:
```
data/translab/
├── scene_01/
│   ├── images/           # Input images
│   ├── sparse/0/         # COLMAP output (cameras.txt, images.txt, points3D.txt)
│   ├── normals/          # StableNormal predictions (per-image .npy)
│   └── masks/            # Transparent region masks (binary .png)
├── scene_02/
│   └── ...
└── scene_08/
```

### DTU Dataset
1. Download DTU dataset following [PGSR](https://github.com/zju3dv/PGSR).
2. Organize as follows:
```
data/dtu_dataset/dtu/
├── scan24/
│   ├── images/
│   └── sparse/0/
├── scan37/
│   └── ...
└── scan122/
```

### Data Preprocessing
```bash
# Generate StableNormal predictions (requires StableNormal installation)
python preprocess/process_normal.py --source_path data/translab/scene_01

# Generate transparent region masks (requires Grounded-SAM2)
# Place output in data/translab/scene_XX/masks/
```

## Training

### TransLab
```bash
# Train all 8 scenes sequentially
python scripts/run_translab.py --out_name reliablegs --gpu_id 0 \
    -d -n --mask_background --eval \
    --normal_folder normals

# Train a single scene
CUDA_VISIBLE_DEVICES=0 python train.py \
    -s data/translab/scene_01 \
    -m output_translab/scene_01/reliablegs \
    -d -n --mask_background --eval \
    --iterations 30000 --delight_iterations 15000 \
    --normal_folder normals --seed 42
```

### DTU
```bash
# Train all 15 scans
python scripts/run_dtu.py --out_name reliablegs --gpu_id 0 \
    -n --eval --iterations 30000

# Train a single scan
CUDA_VISIBLE_DEVICES=0 python train.py \
    -s data/dtu_dataset/dtu/scan24 \
    -m output_dtu/dtu_scan24/reliablegs \
    -n --eval --iterations 30000 --seed 42
```

Key training flags:
- `-d`: Enable delight mode (for TransLab transparent scenes)
- `-n`: Enable normal supervision
- `--mask_background`: Mask background regions (TransLab)
- `--iterations`: Total training iterations (default: 30000)
- `--delight_iterations`: Geometry-phase iterations before appearance (default: 15000)

## Rendering & Mesh Extraction

```bash
# Render test views and extract mesh
python render.py -m output_translab/scene_01/reliablegs \
    -d -n --mask_background --eval \
    --num_cluster 5 --voxel_size 0.002 --max_depth 10.0 \
    --use_transparent_depth

# For DTU
python render.py -m output_dtu/dtu_scan24/reliablegs \
    -n --eval --num_cluster 1 --voxel_size 0.004 --max_depth 10.0
```

The extracted mesh will be saved to `<model_path>/mesh/tsdf_fusion_post.ply`.

## Evaluation

### TransLab (Geometry)
```bash
python scripts/eval_translab/eval.py \
    --data output_translab/scene_01/reliablegs/mesh/tsdf_fusion_post.ply \
    --scan scene_01 --dataset_dir data/translab \
    --mode mesh --downsample_density 0.002 \
    --vis_out_dir output_translab/scene_01/reliablegs/mesh
```

### DTU (Chamfer Distance)
```bash
python scripts/eval_dtu/eval.py \
    --data output_dtu/dtu_scan24/reliablegs/mesh/tsdf_fusion_post.ply \
    --scan_id 24 \
    --dataset_dir data/dtu_dataset \
    --vis_out_dir output_dtu/dtu_scan24/reliablegs/mesh
```

### Image Quality Metrics
```bash
python metrics.py -m output_translab/scene_01/reliablegs
```

## Project Structure

```
├── train.py                 # Main training script
├── render.py                # Rendering and mesh extraction
├── metrics.py               # PSNR/SSIM/LPIPS evaluation
├── arguments/               # Command-line argument definitions
├── gaussian_renderer/       # Differentiable Gaussian rasterizer
├── scene/                   # Scene, camera, and Gaussian model
├── utils/
│   ├── reliability_stats.py # R-map computation (core contribution)
│   ├── mv_normal_utils.py   # Multi-view normal estimation loss
│   ├── loss_utils.py        # Loss functions (L1, SSIM, LNCC)
│   └── ...
├── preprocess/              # Data preprocessing scripts
├── submodules/              # CUDA extensions
│   ├── diff-first-surface-rasterization/
│   └── simple-knn/
└── scripts/                 # Run and evaluation scripts
```

## Core Components

- **R-map** (`utils/reliability_stats.py`): Multi-View Geometric Reliability Map that quantifies per-pixel geometric trustworthiness.
- **MNE Loss** (`utils/mv_normal_utils.py`): Reliability-guided multi-view normal estimation with inverse sampling.
- **Loss Modulation** (`train.py`): Dual-ended weighting for geometric loss, single-ended suppression for NCC loss.
- **Adaptive Pruning** (`scene/gaussian_model.py`): Reliability-guided phantom Gaussian removal.

## License

This project is for non-commercial research use only. See [LICENSE.md](LICENSE.md) for details.
The CUDA rasterizer is based on 3DGS (Inria/MPII license). See `submodules/` for their licenses.
