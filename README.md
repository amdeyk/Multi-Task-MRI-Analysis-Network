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

## License
This project is released under the MIT License.
