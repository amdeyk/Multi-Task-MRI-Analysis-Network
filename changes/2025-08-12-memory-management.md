# Memory management and adaptive batching

- Added `memory_manager.py` providing GPU memory monitoring, adaptive batch sizing
  and basic OOM recovery utilities.
- Extended `config.TrainingConfig` with options for adaptive batching and batch
  size scheduling.
- Integrated the `GPUMemoryManager` into `trainer.py` for runtime monitoring and
  graceful OOM handling.
- Added `memory_optimization.md` documentation outlining memory tuning
  strategies.
