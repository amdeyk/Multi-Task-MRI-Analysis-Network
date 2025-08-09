# Changelog

## [Unreleased]
- Rewrote README with comprehensive technical documentation.
- Initial implementation of modular NumPy-based multi-task MRI network.
- Added training, prediction utilities and demo scripts.
- Included README and unit tests infrastructure.
- Migrated core network operations to PyTorch for accelerated computation.
- Expanded README with advanced architectural documentation.
- Introduced advanced configuration system, optimized attention and KAN modules.
- Added enhanced MRI network architecture, CLI utilities, losses, and data pipeline.
- Implemented full training/evaluation CLI with checkpointing and metrics.
- Added DICOM data loading with preprocessing and metadata handling.
- Replaced placeholder transformer block and KAN layer with production-ready versions.
- Optimized cube splitting with vectorised extraction, adaptive batching and
  overlap-aware reconstruction.
- Implemented efficient differential feature extraction with Sobel gradients
  and caching.
- Added adaptive cube embedding, memory-aware chunking and gradient
  checkpointing options in transformer blocks.
- Unified previously separate network implementations under a configurable
  ``BaseMRINet`` with optional advanced modules and standardised ``(B, C, D, H, W)``
  tensor shapes.
