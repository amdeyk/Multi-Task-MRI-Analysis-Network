# Memory Optimization Strategies

This project includes a lightweight GPU memory manager located in
`memory_manager.py`. Key features include:

- Real‑time monitoring of allocated and reserved memory
- Automatic garbage collection and cache clearing
- Adaptive batch sizing with retry logic on out‑of‑memory (OOM) errors
- Simple model sharding support via `torch.nn.DataParallel`
- Basic leak detection utilities

For best results:

1. Enable adaptive batching in `config.py` by setting
   `training.adaptive_batch = True` and optionally providing
   `batch_size_schedule`.
2. Use the `GPUMemoryManager.optimize_batch_size` helper to determine a safe
   batch size prior to training on new hardware.
3. Monitor logs for memory warnings to catch potential leaks early.
4. When OOM errors occur the manager will automatically reduce the batch size
   and retry. If recovery is not possible the error is propagated for
   visibility.

These tools offer a starting point for more advanced memory optimisation,
including model sharding across multiple GPUs and hybrid CPU/GPU fallbacks.
