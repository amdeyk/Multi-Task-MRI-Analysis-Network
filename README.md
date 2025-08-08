# Multi-Task MRI Analysis Network

This repository provides a compact, modular pipeline for experimenting with
multi-task learning on MRI volumes. The code focuses on clarity and
extensibility and relies only on NumPy so that it can run in very lightweight
Python environments.

## Features
- Differential feature extraction
- Cube-based 3D embeddings
- Residual transformer style mixing
- Kolmogorov–Arnold Network (KAN) inspired heads
- Modular training, prediction and data loading utilities

## Installation
The project has minimal dependencies. NumPy is required for the example
implementation:

```bash
pip install numpy
```

If the optional [KAN](https://github.com/KindXiaoming/kan) package or
`torch` are installed, the heads can seamlessly use them.

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
Unit tests use `pytest`. They automatically skip network checks if PyTorch is
missing, but basic shape checks still run:

```bash
pytest
```

## License
This project is released under the MIT License.
