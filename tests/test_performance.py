"""Lightweight placeholders for performance tests."""

import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

@pytest.fixture
@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def model():
    from config import Config
    from optimized_network import SOTAMRINetwork

    cfg = Config()
    return SOTAMRINetwork(cfg)


@pytest.fixture
@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def sample_input():
    return torch.randn(2, 2, 16, 128, 128)


@pytest.mark.skip(reason="Performance intensive; skipping in CI")
def test_inference_speed(model, sample_input):
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)


@pytest.mark.skip(reason="Performance intensive; skipping in CI")
def test_memory_usage(model, sample_input):
    _ = model(sample_input)


@pytest.mark.skip(reason="Performance intensive; skipping in CI")
def test_gradient_computation(model, sample_input):
    output = model(sample_input)
    loss = sum(o.mean() for o in output.values())
    loss.backward()
