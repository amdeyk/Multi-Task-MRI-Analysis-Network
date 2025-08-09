# Multi-Task MRI Analysis Network: Production-Ready Implementation

## Project Overview

This repository implements a production-grade multi-task learning framework for MRI volume analysis, featuring advanced deep learning techniques including Flash Attention, Kolmogorov-Arnold Networks (KANs), Mixture of Experts (MoE), and sophisticated loss functions. The system provides comprehensive CLI tools, distributed training support, memory management, and medical imaging data pipelines.

## Key Features

- ✅ **Complete Training Pipeline**: Full training, validation, and inference workflows
- ✅ **Medical Imaging Support**: Native DICOM and NIfTI format handling
- ✅ **Distributed Training**: Multi-GPU and multi-node training with automatic setup
- ✅ **Memory Management**: Adaptive batch sizing and GPU memory optimization
- ✅ **Production CLI**: Comprehensive command-line interface with rich output
- ✅ **Experiment Tracking**: Integration with Weights & Biases and MLflow
- ✅ **Model Serving**: FastAPI-based inference server
- ✅ **Medical Compliance**: Metadata handling, anonymization, and validation

## Architecture Overview

### Unified Network Design

The project centers around `mri_network.BaseMRINet`, a configurable backbone that standardizes tensor shapes to `(B, C, D, H, W)` throughout the pipeline. The architecture supports:

- **Lightweight Configuration**: `create_basic_net()` for experiments and prototyping
- **State-of-the-Art Configuration**: `create_sota_net()` with all advanced modules enabled
- **Modular Components**: Optional Flash Attention, MoE, and parallel KAN heads

### Core Innovation Stack

1. **Adaptive Patch Embedding**: Memory-aware sliding window processing with distributed volume handling
2. **Production Transformer Blocks**: Real multi-head attention with gradient checkpointing
3. **Advanced KAN Implementation**: Vectorized B-spline basis functions with learnable knot placement
4. **Multi-Task Loss Framework**: Uncertainty-weighted loss with medical imaging specific losses
5. **Comprehensive Data Pipeline**: DICOM/NIfTI support with preprocessing and metadata management

## Detailed Component Analysis

### 1. Advanced Attention Mechanisms (`attention.py`)

#### Flash Attention Implementation
- **Memory Efficiency**: Uses PyTorch's Scaled Dot Product Attention kernels
- **Hardware Optimization**: Automatic kernel selection (Flash, Memory-Efficient, Math)
- **Production Ready**: Proper dropout and mask handling

#### Sparse Attention Implementation
- **Windowed Processing**: O(n·w) complexity for long sequences
- **Learnable Patterns**: Trainable attention bias for each window

### 2. Enhanced KAN Implementation (`advanced_kan.py`)

#### Optimized KAN Layer
- **Vectorized B-splines**: Cox-de Boor recursion without Python loops
- **Adaptive Basis Selection**: Learnable gates for basis function selection
- **GPU Acceleration**: Efficient batched computation with proper gradient flow

#### Parallel KAN Heads
- **Multi-Task Processing**: Concurrent task execution with CUDA streams
- **CPU Fallback**: ThreadPoolExecutor for CPU-based parallel execution

### 3. Production Network Architecture (`optimized_network.py`)

#### Adaptive Patch Embedding
- **Memory-Aware Processing**: Dynamic chunking based on available GPU memory
- **Distributed Support**: Automatic slice distribution across ranks with gather operations
- **Multi-Scale Features**: Optional multi-resolution processing for better feature extraction

#### Enhanced Transformer Blocks
- **Gradient Checkpointing**: Automatic checkpointing based on memory pressure
- **Mixture of Experts**: Optional conditional computation for efficiency
- **Stochastic Depth**: Layer-level dropout for regularization

### 4. Advanced Loss Functions (`losses.py`)

#### Medical-Specific Losses
- **Dice Loss**: Optimized for segmentation tasks with smooth gradients
- **Boundary Loss**: 3D boundary detection for anatomical edge enhancement
- **Focal Loss**: Addresses class imbalance common in medical imaging
- **Multi-Task Weighting**: Learnable uncertainty-based task balancing

### 5. Production Data Pipeline (`data_pipeline.py`)

#### Medical Imaging Support
- **DICOM Processing**: Native DICOM reading with series detection and grouping
- **NIfTI Support**: Comprehensive NIfTI file handling with metadata extraction
- **Preprocessing Pipeline**: Standardized intensity normalization, resampling, and orientation
- **Data Validation**: Comprehensive quality checks and metadata validation

#### Advanced Features
- **Metadata Management**: Extraction, standardization, and anonymization
- **Distributed Sampling**: DistributedSampler integration for multi-GPU training
- **Caching System**: LRU cache for frequently accessed volumes

### 6. Training Framework (`trainer.py`)

#### Production Training
- **Automatic Mixed Precision**: Configurable AMP with overflow detection
- **Memory Management**: Integration with GPU memory manager for OOM recovery
- **Comprehensive Logging**: Structured logging with memory and performance metrics
- **Checkpoint Management**: Save/resume functionality with best model tracking

#### Training Manager
- **Early Stopping**: Configurable patience-based early stopping
- **Learning Rate Scheduling**: Integrated scheduler support
- **Experiment Tracking**: Automatic logging to W&B or MLflow
- **Distributed Training**: Seamless multi-GPU and multi-node training

### 7. Memory Management (`memory_manager.py`)

#### GPU Memory Optimization
- **Real-time Monitoring**: Live memory usage tracking and reporting
- **Adaptive Batch Sizing**: Automatic batch size optimization with OOM recovery
- **Memory Leak Detection**: Basic leak detection and alerting
- **Model Sharding**: Simple DataParallel integration for multi-GPU setups

### 8. Distributed Training (`distributed_training.py`)

#### Multi-GPU Support
- **Automatic Initialization**: Seamless process group setup
- **Device Management**: Automatic device assignment and model wrapping
- **CLI Integration**: Distributed training through command-line interface

### 9. Command Line Interface (`cli.py`)

#### Production CLI
- **Training Command**: Full training pipeline with distributed support
- **Inference Command**: Batch prediction with multiple output formats
- **Evaluation Command**: Comprehensive metrics computation including medical-specific metrics
- **Configuration Wizard**: Interactive setup for new projects

### 10. Model Serving (`serving.py`)

#### FastAPI Server
- **REST API**: Production-ready inference endpoints
- **Batch Processing**: Efficient batch inference capabilities
- **Error Handling**: Comprehensive error handling and logging
- **Health Checks**: Model loading and status monitoring

## Installation and Setup

### Core Dependencies
```bash
# Essential packages
torch>=2.0.0          # PyTorch deep learning framework
torchvision>=0.15.0   # Computer vision utilities
einops>=0.7.0         # Tensor manipulation library
albumentations>=1.3.0 # Advanced augmentations
nibabel>=5.0.0        # Medical imaging I/O
rich>=13.0.0          # Terminal formatting
click>=8.0.0          # CLI framework
fastapi>=0.68.0       # Model serving
```

### Medical Imaging Dependencies
```bash
# Medical imaging support
pydicom>=2.3.0        # DICOM file handling
SimpleITK>=2.2.0      # Medical image processing
monai>=1.0.0          # Medical imaging transforms
```

### Optional Dependencies
```bash
# Enhanced functionality
wandb>=0.15.0         # Experiment tracking
mlflow>=2.0.0         # Alternative experiment tracking
kan>=0.2.0            # Enhanced KAN implementation
```

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/your-org/mri-kan.git
cd mri-kan

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install torch torchvision albumentations nibabel rich click fastapi
```

### 2. Configuration Setup
```bash
# Interactive configuration wizard
python -m cli init
```

### 3. Training
```bash
# Single GPU training
python -m cli train -d /path/to/data -c config.yaml

# Multi-GPU training
python -m cli train -d /path/to/data -c config.yaml -g 0 1 2 3

# Distributed training with torchrun
torchrun --nproc_per_node=4 -m cli train -d /path/to/data -c config.yaml
```

### 4. Inference
```bash
# Single file prediction
python -m cli predict -m checkpoints/best.pt -i scan.nii.gz -o predictions/

# Batch prediction
python -m cli predict -m checkpoints/best.pt -i /path/to/scans/ -o predictions/
```

### 5. Evaluation
```bash
# Compute metrics on test set
python -m cli evaluate -m checkpoints/best.pt -d /path/to/test_data/
```

### 6. Model Serving
```bash
# Start inference server
uvicorn serving:app --host 0.0.0.0 --port 8000
```

## Data Format Requirements

### Input Data Structure
```
data/
├── train/
│   ├── patient_001/
│   │   ├── t1.nii.gz
│   │   ├── t2.nii.gz
│   │   └── mask.nii.gz
│   └── patient_002/
│       ├── series_001/  # DICOM directory
│       │   ├── *.dcm
│       └── RTSTRUCT*.dcm
└── val/
    └── ...
```

### Supported Formats
- **NIfTI**: `.nii`, `.nii.gz` files with optional mask files
- **DICOM**: Complete DICOM series with automatic series detection
- **Metadata**: Automatic extraction and standardization from DICOM headers

## Configuration

### Model Configuration
```yaml
model:
  embed_dim: 128
  num_heads: 8
  num_layers: 6
  use_flash_attention: true
  use_moe: false
  num_experts: 4
```

### Training Configuration
```yaml
training:
  batch_size: 8
  adaptive_batch: true
  learning_rate: 1e-4
  epochs: 100
  mixed_precision: true
  gradient_clip: 1.0
```

### Data Configuration
```yaml
data:
  patch_size: [128, 128, 128]
  overlap: 0.25
  augmentation: true
  normalize: true
```

## Performance Benchmarks

### Hardware Requirements

#### Minimum Configuration
- **GPU**: 8GB VRAM (RTX 3070/A4000)
- **RAM**: 32GB system memory
- **Storage**: 100GB NVMe SSD

#### Recommended Configuration
- **GPU**: 24GB VRAM (RTX 4090/A6000)
- **RAM**: 128GB system memory
- **Storage**: 1TB NVMe SSD

#### Production Configuration
- **GPU**: Multi-GPU setup (4×A100 40GB)
- **RAM**: 256GB+ system memory
- **Storage**: Multi-TB NVMe RAID

### Scaling Guidelines

#### Model Variants
- **Basic**: 15M parameters, embed_dim=64, 4 layers
- **Standard**: 32M parameters, embed_dim=128, 6 layers
- **Large**: 85M parameters, embed_dim=256, 8 layers
- **XL**: 200M parameters, embed_dim=512, 12 layers

## Medical AI Considerations

### Data Privacy and Compliance
- **Automatic Anonymization**: Patient ID and demographic data anonymization
- **Metadata Validation**: Comprehensive DICOM metadata validation
- **Audit Logging**: Complete audit trails for all data processing

### Clinical Validation
- **Medical Metrics**: Dice coefficient, Hausdorff distance, sensitivity, specificity
- **Statistical Testing**: Significance testing across demographic groups
- **Bias Detection**: Automated bias detection in model predictions

## Testing and Validation

### Unit Testing
```bash
# Run test suite
pytest tests/

# Run with coverage
pytest tests/ --cov=.
```

### Performance Testing
```bash
# Memory profiling
python -m cli train --profile -d data/

# Distributed testing
torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py
```

## Contributing

### Development Setup
```bash
# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run linting
black . && isort . && flake8 .
```

### Code Quality
- **Type Hints**: All functions include comprehensive type annotations
- **Documentation**: Detailed docstrings following NumPy convention
- **Testing**: Comprehensive unit and integration test coverage
- **Linting**: Black, isort, and flake8 for code formatting and quality

## Known Limitations

### Current Constraints
- **Memory Usage**: Large volumes (>512³) require distributed processing
- **DICOM Variants**: Some non-standard DICOM files may not be supported
- **Platform Support**: Optimized for Linux with CUDA; limited Windows/macOS testing

### Future Improvements
- **Model Quantization**: INT8 quantization for deployment
- **ONNX Export**: Cross-platform model deployment
- **TensorRT Optimization**: Inference acceleration for production
- **Federated Learning**: Multi-institutional training support

## License and Citation

This project is released under the MIT License. If you use this code in your research, please cite:

```bibtex
@software{mri_kan_2024,
  title={MRI-KAN: Multi-Task MRI Analysis with Kolmogorov-Arnold Networks},
  author=Ambarish Dey,
  year=2025,
  url={https://github.com/amdeyk/Multi-Task-MRI-Analysis-Network}
}
```

## Support and Documentation

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive API documentation in `/docs/`
- **Examples**: Tutorial notebooks in `/examples/`
- **Community**: Discord server for discussions and support

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and recent updates.

---

**Note**: This is a production-ready implementation with comprehensive testing and validation. For research use, ensure compliance with your institution's medical data handling policies.
