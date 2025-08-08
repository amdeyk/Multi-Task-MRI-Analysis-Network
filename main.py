import numpy as np
from multitask_net import MultiTaskMRINet
from data_loader import get_samples
from trainer import train_one_epoch, validate
from predict import predict


def main() -> None:
    model = MultiTaskMRINet(
        in_channels=2,
        cube_size=8,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        n_tasks=4,
    )
    train_data = get_samples(4)
    val_data = get_samples(2)
    for epoch in range(2):
        loss = train_one_epoch(model, train_data)
        val = validate(model, val_data)
        print(f"Epoch {epoch+1}: train_loss={loss:.4f} val_loss={val:.4f}")
    sample = get_samples(1)[0]
    preds = predict(model, sample["mri"][None])
    print("Prediction keys:", list(preds.keys()))


if __name__ == "__main__":
    main()
