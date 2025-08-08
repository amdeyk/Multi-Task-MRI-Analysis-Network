import numpy as np


def predict(model, mri_vol: np.ndarray) -> dict[str, np.ndarray]:
    """Run a forward pass and apply basic activations."""
    outputs = model.forward(mri_vol)
    outputs["segmentation"] = softmax(outputs["segmentation"], axis=-1)
    outputs["classification"] = sigmoid(outputs["classification"])
    outputs["edge"] = sigmoid(outputs["edge"])
    outputs["tumor"] = softmax(outputs["tumor"], axis=-1)
    return outputs


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
