# Multi-Task MRI Analysis Network

This repository provides a compact, modular pipeline for experimenting with
multi-task learning on MRI volumes. The code focuses on clarity and
extensibility and now utilises PyTorch tensors for accelerated computation
while remaining lightweight.

## Features
- Differential feature extraction
- Cube-based 3D embeddings
- Residual transformer style mixing
- Kolmogorov–Arnold Network (KAN) inspired heads
- Modular training, prediction and data loading utilities

## Installation
The project has minimal dependencies. PyTorch provides the tensor backend:

```bash
pip install torch
```

If the optional [KAN](https://github.com/KindXiaoming/kan) package is
installed the heads can seamlessly use it.

## Usage
Run a quick end-to-end demonstration:

```bash
python run_demo.py
```

For a miniature training/evaluation loop and a prediction example:

```bash
python main.py
```

## Testing
Unit tests use `pytest`:

```bash
pytest
```

## Multi-Task MRI Analysis Network: Advanced Documentation

### Overview

This project implements a sophisticated multi-task learning pipeline for MRI volume analysis, combining differential feature extraction, 3D cube-based embeddings, residual transformer architectures, and Kolmogorov-Arnold Network (KAN) inspired heads. The system is designed for simultaneous prediction of multiple MRI-related tasks including segmentation, classification, edge detection, and tumor classification.

### Architectural Components

#### 1. Differential Feature Extraction (`differential.py`)

The `DifferentialFeatureExtractor` implements a novel approach to enhance volumetric features by computing temporal and spatial differentials:

**Key Concepts:**
- **Slice Differentials**: Captures inter-slice variations to detect anatomical transitions
- **Channel Differentials**: Exploits contrast differences between MRI sequences (T1, T2, FLAIR, etc.)
- **Spatial Gradients**: Optional gradient computation for edge enhancement
- **Feature Concatenation**: Combines original and differential features for richer representations

**Mathematical Foundation:**
```
Output = Concat[X, ∇_slice(X), ∇_channel(X), ∇_x(X), ∇_y(X)]
```

This approach significantly increases the feature dimensionality while providing the network with explicit differential information that would otherwise need to be learned implicitly.

#### 2. 3D Cube Embedding (`cube_embed.py`)

The `CubeSplitter3D` module transforms volumetric data into a sequence of embedded cube representations:

**Architecture Details:**
- **Non-overlapping Cubes**: Divides volumes into `cube_size³` voxel cubes
- **Linear Projection**: Each cube is flattened and projected to `embed_dim` dimensions
- **Spatial Locality**: Preserves local 3D spatial relationships within each cube
- **Scalability**: Handles variable input sizes through adaptive cube extraction

**Design Rationale:**
- Enables transformer-style processing of 3D volumes
- Reduces computational complexity compared to full 3D convolutions
- Maintains spatial coherence through cube-level processing
- Facilitates parallel processing of spatial regions

#### 3. Residual Transformer Architecture (`residual_transformer.py`)

The transformer blocks implement a simplified but effective mixing strategy:

**Current Implementation:**
- Simple residual connections with previous layer outputs
- Lightweight design for rapid prototyping
- Maintains gradient flow through deep networks

**Potential Extensions:**
- Multi-head self-attention mechanisms
- Feed-forward networks with non-linear activations
- Layer normalization and dropout for regularization
- Positional encoding for spatial awareness

#### 4. KAN-Inspired Task Heads (`sota_kan.py`)

The output heads leverage Kolmogorov-Arnold Networks when available:

**KAN Advantages:**
- **Function Approximation**: KANs can represent complex multivariate functions more efficiently
- **Interpretability**: Learnable activation functions provide insights into decision boundaries
- **Flexibility**: Adaptive activation functions vs. fixed activations in traditional networks
- **Graceful Fallback**: Automatically uses standard linear layers if KAN package unavailable

### Multi-Task Learning Strategy

#### Task Definitions

1. **Segmentation**: Binary/multi-class voxel-level tissue classification
2. **Classification**: Volume-level diagnostic classification
3. **Edge Detection**: Boundary detection for anatomical structures
4. **Tumor Classification**: Multi-class tumor type identification

#### Task-Specific Considerations

**Segmentation Head:**
- Outputs per-cube predictions that need spatial reconstruction
- Requires careful handling of cube boundaries
- Benefits from high spatial resolution features

**Classification Head:**
- Uses global average pooling across all cubes
- Captures volume-level patterns and statistics
- Less sensitive to local spatial details

**Edge Detection:**
- Leverages differential features heavily
- Benefits from spatial gradient information
- Requires fine-grained spatial predictions

**Tumor Classification:**
- Combines local (cube-level) and global (volume-level) features
- Multiple classes require careful loss balancing
- Critical for clinical decision support

### Data Pipeline Architecture

#### Synthetic Data Generation (`data_loader.py`)

The `DummyMRIDataset` provides realistic data simulation:

**Features:**
- Multi-contrast MRI volumes (T1, T2, FLAIR simulation)
- Realistic dimensionality (64×64×16 with 2 contrasts)
- Multiple ground truth tasks with appropriate data types
- Consistent batch generation for reproducible experiments

**Usage Patterns:**
- Rapid prototyping without large datasets
- Algorithm validation and debugging
- Performance benchmarking and profiling

### Training and Evaluation Framework

#### Training Pipeline (`trainer.py`)

**Current Implementation:**
- Simplified training loop for demonstration
- Basic forward pass and loss computation
- Separate training and validation phases

**Production Considerations:**
- Multi-task loss weighting strategies
- Learning rate scheduling
- Gradient clipping for stability
- Mixed precision training for efficiency
- Checkpoint saving and resuming

#### Prediction Interface (`predict.py`)

**Post-processing Strategy:**
- Task-appropriate activation functions
- Softmax for multi-class outputs (segmentation, tumor)
- Sigmoid for binary classification tasks
- Consistent output formatting across tasks

### Technical Implementation Details

#### PyTorch Integration

**Design Decisions:**
- Full PyTorch tensor operations for GPU acceleration
- Automatic differentiation support throughout
- Memory-efficient tensor operations
- Native PyTorch module integration

**Performance Optimizations:**
- Vectorized cube extraction and embedding
- Batch-wise processing where possible
- Efficient tensor concatenation and reshaping
- GPU-friendly memory layouts

#### Memory Management

**Cube Processing:**
- Dynamic cube extraction avoids pre-allocation
- Efficient tensor stacking and reshaping
- Handles variable input sizes gracefully
- Memory-efficient gradient computation

**Differential Features:**
- In-place operations where possible
- Careful tensor concatenation to avoid copies
- Gradient-friendly implementations

### Advanced Usage Patterns

#### Custom Task Integration

To add new tasks:

1. Define new head in `multitask_net.py`
2. Add corresponding output processing in `predict.py`
3. Update training loop with appropriate loss functions
4. Modify data loader for new ground truth labels

#### Hyperparameter Tuning

**Critical Parameters:**
- `cube_size`: Balance between spatial detail and computational efficiency
- `embed_dim`: Feature representation capacity
- `num_heads`: Attention mechanism complexity
- `num_layers`: Network depth and representational capacity

**Scaling Guidelines:**
- Larger volumes → larger cube sizes or more cubes
- More complex tasks → higher embedding dimensions
- Deeper understanding → more transformer layers
- Better hardware → larger batch sizes and model capacity

#### Performance Optimization

**GPU Acceleration:**
- Ensure all tensors are on the same device
- Use appropriate batch sizes for GPU memory
- Consider mixed precision training for large models
- Profile memory usage and computational bottlenecks

**Distributed Training:**
- Model parallelism for very large networks
- Data parallelism for larger datasets
- Gradient accumulation for effective large batch training

### Research Extensions

#### Potential Improvements

1. **Enhanced Transformer Architecture**
   - Full multi-head attention implementation
   - Positional encoding for spatial relationships
   - Layer normalization and advanced regularization

2. **Advanced Cube Processing**
   - Overlapping cubes for better boundary handling
   - Hierarchical cube sizes for multi-scale features
   - Learned cube selection strategies

3. **Improved Multi-Task Learning**
   - Uncertainty-weighted loss functions
   - Task-specific feature branches
   - Cross-task attention mechanisms

4. **Clinical Integration**
   - DICOM file format support
   - Real-world data preprocessing pipelines
   - Validation on clinical datasets

### Dependencies and Requirements

#### Core Dependencies
- PyTorch: Tensor operations and neural network modules
- NumPy: Numerical computations and array operations

#### Optional Dependencies
- KAN package: Enhanced function approximation capabilities
- pytest: Unit testing framework

#### System Requirements
- GPU recommended for larger volumes
- Sufficient RAM for batch processing
- Python 3.8+ for type annotation support

### Testing and Validation

The project includes comprehensive unit tests:
- Forward pass shape validation
- Multi-task output verification
- Integration testing across components
- Performance benchmarking capabilities

### Conclusion

This multi-task MRI analysis network represents a modern approach to medical image analysis, combining classical computer vision techniques with state-of-the-art deep learning architectures. The modular design facilitates rapid experimentation while maintaining the flexibility to scale to production medical imaging applications.

The integration of differential features, 3D cube embeddings, and KAN-inspired heads creates a unique pipeline optimized for the specific challenges of multi-contrast MRI analysis. The codebase serves as both a research platform and a foundation for clinical deployment.

## License
This project is released under the MIT License.
