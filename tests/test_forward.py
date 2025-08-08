import pathlib, sys
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from multitask_net import MultiTaskMRINet


def test_forward_shapes() -> None:
    model = MultiTaskMRINet(
        in_channels=2,
        cube_size=8,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        n_tasks=4,
    )
    mri_vol = torch.randn(1, 2, 16, 64, 64)
    outputs = model(mri_vol)
    assert set(outputs.keys()) == {"segmentation", "classification", "edge", "tumor"}
    assert outputs["classification"].shape[0] == 1
