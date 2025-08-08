import numpy as np

try:  # pragma: no cover - optional dependency
    from kan import KANLinear  # type: ignore
except Exception:  # pragma: no cover - fall back to simple linear layer
    class KANLinear:
        def __init__(self, in_dim: int, out_dim: int) -> None:
            self.weight = np.random.randn(in_dim, out_dim).astype(np.float32)
            self.bias = np.zeros(out_dim, dtype=np.float32)

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return x @ self.weight + self.bias

class SOTAKANHead:
    """Simple wrapper around ``KANLinear`` or a NumPy fallback."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.kan = KANLinear(in_dim, out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        b, n, d = x.shape
        flat = x.reshape(-1, d)
        out = self.kan(flat)
        return out.reshape(b, n, -1)
