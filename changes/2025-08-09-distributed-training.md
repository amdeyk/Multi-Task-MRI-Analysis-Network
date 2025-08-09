# 2025-08-09 Distributed training and AMP overhaul

- Introduced `distributed_training.py` with utilities for initialising and cleaning up PyTorch DDP sessions. The helpers also expose device selection and model wrapping functions for multi-GPU and multi-node setups.
- Reworked the CLI `train` command to launch in distributed contexts, automatically wrap models, and restrict console output to the main process. Data loading now uses `DistributedSampler` when appropriate.
- Added automatic mixed precision controls to the `Trainer` with configurable dtype, overflow detection and memory logging.
- Updated data pipeline to construct samplers for distributed training.
- Replaced cube extraction in `optimized_network` with an adaptive sliding-window embedding that chunks volumes based on free memory and gathers results across ranks.
