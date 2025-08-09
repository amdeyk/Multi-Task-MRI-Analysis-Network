import gc
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch


@dataclass
class GPUMemoryManager:
    """Utility for monitoring and controlling GPU memory.

    The manager exposes helpers for real-time memory monitoring, adaptive
    batch-size tuning and basic out-of-memory recovery strategies.  The
    implementation is intentionally lightweight – heavier integration points
    can extend this class.
    """

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_utilization: float = 0.9
    min_batch_size: int = 1
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    current_batch_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Monitoring utilities
    # ------------------------------------------------------------------
    def monitor(self) -> Dict[str, int]:
        """Return current memory statistics for the active device."""
        if not torch.cuda.is_available():  # pragma: no cover - depends on hardware
            return {}
        total = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        free = total - allocated
        self.logger.debug(
            "GPU memory - total=%d allocated=%d reserved=%d free=%d", total, allocated, reserved, free
        )
        return {
            "total": total,
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
        }

    def empty_cache(self) -> None:
        """Run garbage collection and free cached CUDA memory."""
        gc.collect()
        if torch.cuda.is_available():  # pragma: no cover - hardware specific
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Adaptive batch sizing
    # ------------------------------------------------------------------
    def optimize_batch_size(self, trial_fn: Callable[[int], None], initial_bs: int) -> int:
        """Find the largest batch size that does not trigger OOM.

        Args:
            trial_fn: callable that performs a forward/backward pass given a
                batch size. It should raise ``RuntimeError`` on OOM.
            initial_bs: starting batch size to try.
        """
        bs = initial_bs
        self.current_batch_size = bs
        while bs >= self.min_batch_size:
            try:
                trial_fn(bs)
                break
            except RuntimeError as exc:  # pragma: no cover - hardware specific
                if "out of memory" in str(exc).lower():
                    self.logger.warning("OOM at batch size %d; reducing", bs)
                    self.empty_cache()
                    bs //= 2
                    self.current_batch_size = bs
                else:
                    raise
        return self.current_batch_size or bs

    # ------------------------------------------------------------------
    # OOM recovery utilities
    # ------------------------------------------------------------------
    def handle_oom(self, exc: RuntimeError) -> bool:
        """Attempt to recover from an OOM error.

        Returns ``True`` if a retry should be attempted with a smaller batch
        size, otherwise ``False`` to propagate the error.
        """
        if "out of memory" not in str(exc).lower():
            return False
        self.logger.error("Out of memory: %s", exc)
        self.empty_cache()
        if self.current_batch_size and self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            self.logger.info("Retrying with batch size %d", self.current_batch_size)
            return True
        self.logger.info("Attempting CPU fallback")
        return False

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------
    def shard_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Enable simple data-parallel sharding across multiple GPUs."""
        if torch.cuda.device_count() > 1:  # pragma: no cover - multi-GPU specific
            self.logger.info("Sharding model across %d GPUs", torch.cuda.device_count())
            return torch.nn.DataParallel(model)
        return model

    def load_model(self, path: str, **kwargs) -> torch.nn.Module:
        """Load a model checkpoint with minimal memory usage."""
        map_loc = kwargs.pop("map_location", self.device)
        return torch.load(path, map_location=map_loc, **kwargs)

    # ------------------------------------------------------------------
    # Debugging helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> int:
        """Return the currently allocated memory snapshot."""
        if not torch.cuda.is_available():  # pragma: no cover - hardware specific
            return 0
        return torch.cuda.memory_allocated(self.device)

    def detect_leak(self, before: int, after: int, tolerance: int = 10_000_000) -> None:
        """Simple memory leak detection based on allocation difference."""
        if after - before > tolerance:  # pragma: no cover - diagnostic utility
            self.logger.warning("Potential memory leak detected: %d bytes", after - before)
