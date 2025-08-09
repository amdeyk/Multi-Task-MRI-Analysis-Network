"""Command line interface for the MRI-KAN project."""

from pathlib import Path

import click
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import track
from rich.table import Table
import yaml
import numpy as np

from config import Config
from optimized_network import SOTAMRINetwork
from data_pipeline import get_dataloader, MRIDataset
from losses import MultiTaskLoss
from trainer import train_one_epoch, validate

try:  # pragma: no cover - optional metrics deps
    from scipy import stats
    from scipy.spatial.distance import directed_hausdorff
except Exception:  # noqa: S110
    stats = None
    directed_hausdorff = None

try:
    import nibabel as nib
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - runtime import
    nib = None
    ToTensorV2 = None
    DataLoader = None

console = Console()


@click.group()
@click.version_option(version="2.0.0")
def cli() -> None:
    """MRI-KAN: Advanced MRI Analysis Network"""
    pass


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Configuration file path")
@click.option("--data", "-d", required=True, help="Training data directory")
@click.option("--output", "-o", default="./checkpoints", help="Output directory")
@click.option("--resume", "-r", help="Resume from checkpoint")
@click.option("--gpu", "-g", multiple=True, type=int, help="GPU IDs to use")
@click.option("--mixed-precision/--no-mixed-precision", default=True, help="Use mixed precision")
@click.option("--profile/--no-profile", default=False, help="Enable profiling")
def train(config, data, output, resume, gpu, mixed_precision, profile) -> None:
    """Train the MRI-KAN model."""
    console.print("[bold green]MRI-KAN Training Pipeline[/bold green]")
    cfg = Config.from_yaml(config)
    cfg.data.data_root = Path(data)
    cfg.training.mixed_precision = mixed_precision
    devices = [f"cuda:{i}" for i in gpu] if gpu else (["cuda"] if torch.cuda.is_available() else ["cpu"])
    device = torch.device(devices[0])
    with console.status("[bold blue]Initializing model..."):
        model = SOTAMRINetwork(cfg)
        if len(devices) > 1:
            model = nn.DataParallel(model, device_ids=[int(d.split(":")[1]) for d in devices])
        model = model.to(device)
    console.print(
        f"[green]\u2713[/green] Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )

    train_loader = get_dataloader(cfg, "train")
    val_loader = get_dataloader(cfg, "val")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs)
    loss_fn = MultiTaskLoss(cfg.loss)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", best_loss)
    patience = 10
    epochs_no_improve = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            scheduler,
            cfg.training.gradient_clip,
        )
        val_metrics = validate(model, val_loader, loss_fn, device)
        console.print(
            f"Epoch {epoch + 1}/{cfg.training.epochs}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f}"
        )
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        }
        torch.save(state, out_path / f"epoch_{epoch}.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(state, out_path / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                console.print("[yellow]Early stopping triggered[/yellow]")
                break


@cli.command()
@click.option("--model", "-m", required=True, help="Model checkpoint path")
@click.option("--input", "-i", required=True, help="Input MRI file or directory")
@click.option("--output", "-o", default="./predictions", help="Output directory")
@click.option("--batch-size", "-b", default=1, help="Batch size for inference")
@click.option("--visualize/--no-visualize", default=True, help="Generate visualizations")
@click.option("--format", type=click.Choice(["nifti", "numpy", "dicom"]), default="nifti")
def predict(model, input, output, batch_size, visualize, format) -> None:
    """Run inference on MRI data."""
    console.print("[bold blue]MRI-KAN Inference Pipeline[/bold blue]")
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SOTAMRINetwork(cfg)
    ckpt = torch.load(model, map_location=device)
    net.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    net.to(device)
    net.eval()
    inp = Path(input)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = ToTensorV2() if ToTensorV2 else None
    if inp.is_dir():
        dataset = MRIDataset(str(inp), transform=transform)
    else:
        dataset = MRIDataset(str(inp.parent), transform=transform)
        dataset.files = [inp]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for idx, batch in enumerate(loader):
        mri = batch["mri"].to(device)
        with torch.no_grad():
            outputs = net(mri)
        pred = outputs["segmentation"].argmax(dim=1).cpu().numpy()
        paths = dataset.files[idx * batch_size : idx * batch_size + len(pred)]
        for pth, pr in zip(paths, pred):
            out_path = out_dir / f"{Path(pth).stem}_pred"
            if format == "numpy" or nib is None:
                np.save(out_path.with_suffix(".npy"), pr)
            else:
                nib.save(nib.Nifti1Image(pr, np.eye(4)), out_path.with_suffix(".nii.gz"))
    console.print("[green]Inference complete[/green]")


@cli.command()
@click.option("--model", "-m", required=True, help="Model checkpoint")
@click.option("--data", "-d", required=True, help="Test data directory")
@click.option("--metrics/--no-metrics", default=True, help="Compute metrics")
@click.option("--save-results/--no-save-results", default=True, help="Save evaluation results")
def evaluate(model, data, metrics, save_results) -> None:
    """Evaluate model performance and compute metrics."""
    console.print("[bold cyan]MRI-KAN Evaluation[/bold cyan]")
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SOTAMRINetwork(cfg)
    ckpt = torch.load(model, map_location=device)
    net.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    net.to(device)
    net.eval()
    transform = ToTensorV2() if ToTensorV2 else None
    dataset = MRIDataset(str(data), transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    for batch in loader:
        mri = batch["mri"].to(device)
        seg = batch["seg"].cpu().numpy()
        meta_list = batch.get("meta", [{}])
        meta = meta_list[0] if isinstance(meta_list, list) else meta_list
        with torch.no_grad():
            outputs = net(mri)
        pred = outputs["segmentation"].argmax(dim=1).cpu().numpy()
        if metrics:
            dice = 2 * np.sum((pred > 0) & (seg > 0)) / (np.sum(pred > 0) + np.sum(seg > 0) + 1e-8)
            if directed_hausdorff is not None:
                pts_p = np.argwhere(pred > 0)
                pts_t = np.argwhere(seg > 0)
                if len(pts_p) and len(pts_t):
                    hd = max(
                        directed_hausdorff(pts_p, pts_t)[0],
                        directed_hausdorff(pts_t, pts_p)[0],
                    )
                else:
                    hd = float("inf")
            else:
                hd = float("nan")
            results.append({"dice": float(dice), "hd95": float(hd), "sex": meta.get("sex", "U")})
    if metrics and results:
        mean_dice = float(np.mean([r["dice"] for r in results]))
        mean_hd = float(np.mean([r["hd95"] for r in results]))
        console.print(f"Mean Dice: {mean_dice:.4f}, Mean HD95: {mean_hd:.2f}")
        if stats is not None:
            males = [r["dice"] for r in results if r["sex"] == "M"]
            females = [r["dice"] for r in results if r["sex"] == "F"]
            if males and females:
                _, p_val = stats.ttest_ind(males, females, equal_var=False)
                console.print(f"Sex t-test p-value: {p_val:.4f}")
        if save_results:
            out_file = Path(data) / "evaluation.npy"
            np.save(out_file, results)
            console.print(f"[green]Saved metrics to {out_file}[/green]")


@cli.command()
def init() -> None:
    """Interactive configuration wizard."""
    console.print("[bold magenta]MRI-KAN Configuration Wizard[/bold magenta]")
    config = {"experiment_name": click.prompt("Experiment name", default="mri_kan_experiment")}
    config["model"] = {
        "embed_dim": click.prompt("Embedding dimension", type=int, default=128),
        "num_heads": click.prompt("Number of attention heads", type=int, default=8),
        "num_layers": click.prompt("Number of transformer layers", type=int, default=6),
        "use_flash_attention": click.confirm("Use Flash Attention?", default=True),
        "use_moe": click.confirm("Use Mixture of Experts?", default=False),
    }
    config["training"] = {
        "batch_size": click.prompt("Batch size", type=int, default=8),
        "learning_rate": click.prompt("Learning rate", type=float, default=1e-4),
        "epochs": click.prompt("Number of epochs", type=int, default=100),
        "mixed_precision": click.confirm("Use mixed precision?", default=True),
    }
    save_path = Path("config.yaml")
    with open(save_path, "w") as f:
        yaml.dump(config, f)
    console.print(f"[green]\u2713 Configuration saved to {save_path}[/green]")


if __name__ == "__main__":
    cli()
