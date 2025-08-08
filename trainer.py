import numpy as np
from typing import Sequence


def train_one_epoch(model, data: Sequence[dict[str, np.ndarray]]) -> float:
    """Very small dummy training loop used for examples."""
    loss = 0.0
    for batch in data:
        out = model.forward(batch["mri"][None])
        loss += float(np.mean(out["classification"]))
    return loss / max(len(data), 1)


def validate(model, data: Sequence[dict[str, np.ndarray]]) -> float:
    """Dummy validation pass."""
    return train_one_epoch(model, data)
