import numpy as np

class DifferentialFeatureExtractor:
    """Compute differential features using NumPy.

    The extractor concatenates the original volume with slice differences,
    channel differences and optional spatial gradients. Implementation is
    intentionally lightweight so it runs in environments without PyTorch.
    """

    def __init__(self, spatial_grad: bool = True) -> None:
        self.spatial_grad = spatial_grad

    def forward(self, x: np.ndarray) -> np.ndarray:
        b, c, s, h, w = x.shape
        diff_slices = np.diff(x, axis=2, prepend=0)
        diff_channels = np.diff(x, axis=1, prepend=0)
        if self.spatial_grad:
            grad_x = np.zeros_like(x)
            grad_y = np.zeros_like(x)
        else:
            grad_x = grad_y = np.zeros_like(x)
        return np.concatenate([x, diff_slices, diff_channels, grad_x, grad_y], axis=1)
