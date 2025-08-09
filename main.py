from pathlib import Path

import torch

from multitask_net import MultiTaskMRINet
from data_loader import get_samples
from losses import MultiTaskLoss
from trainer import Trainer, TrainingManager
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss_fn = MultiTaskLoss({})
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        scaler=None,
        scheduler=scheduler,
    )
    manager = TrainingManager(
        trainer=trainer,
        train_loader=train_data,
        val_loader=val_data,
        epochs=2,
        out_dir=Path("./"),
        patience=2,
    )
    manager.fit()
    sample = get_samples(1)[0]
    preds = predict(model, sample["mri"][None])
    print("Prediction keys:", list(preds.keys()))


if __name__ == "__main__":
    main()
