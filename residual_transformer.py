import numpy as np

class ResidualTransformerBlock:
    """Placeholder transformer block using NumPy operations.

    It merely mixes the current input with the previous residual to keep the
    example lightweight.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2, dropout: float = 0.1) -> None:  # noqa: D401
        self.dim = dim

    def forward(self, x: np.ndarray, prev_residual: np.ndarray | None = None) -> np.ndarray:
        if prev_residual is None:
            return x
        return 0.5 * (x + prev_residual)
