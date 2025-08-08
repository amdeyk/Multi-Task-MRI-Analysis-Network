import numpy as np
from multitask_net import MultiTaskMRINet


def main() -> None:
    mri_vol = np.random.randn(1, 2, 16, 64, 64).astype(np.float32)
    model = MultiTaskMRINet(
        in_channels=2,
        cube_size=8,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        n_tasks=4,
        face_embed=True,
    )
    outputs = model.forward(mri_vol)
    for key, value in outputs.items():
        print(f"{key}: shape {value.shape}")


if __name__ == "__main__":
    main()
