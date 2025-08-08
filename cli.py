"""Command line interface for the MRI-KAN project."""

from pathlib import Path

import click
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import track
from rich.table import Table
import yaml

from config import Config
from optimized_network import SOTAMRINetwork

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
    with console.status("[bold blue]Initializing model..."):
        model = SOTAMRINetwork(cfg)
        if len(devices) > 1:
            model = nn.DataParallel(model, device_ids=[int(d.split(":")[1]) for d in devices])
        model = model.to(devices[0])
    console.print(f"[green]\u2713[/green] Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    console.print("Training loop not implemented in this demo.")


@cli.command()
@click.option("--model", "-m", required=True, help="Model checkpoint path")
@click.option("--input", "-i", required=True, help="Input MRI file or directory")
@click.option("--output", "-o", default="./predictions", help="Output directory")
@click.option("--batch-size", "-b", default=1, help="Batch size for inference")
@click.option("--visualize/--no-visualize", default=True, help="Generate visualizations")
@click.option("--format", type=click.Choice(["nifti", "numpy", "dicom"]), default="nifti")
def predict(model, input, output, batch_size, visualize, format) -> None:
    """Run inference on MRI data (stub)."""
    console.print("[bold blue]MRI-KAN Inference Pipeline[/bold blue]")
    console.print("Inference pipeline not implemented in this demo.")


@cli.command()
@click.option("--model", "-m", required=True, help="Model checkpoint")
@click.option("--data", "-d", required=True, help="Test data directory")
@click.option("--metrics/--no-metrics", default=True, help="Compute metrics")
@click.option("--save-results/--no-save-results", default=True, help="Save evaluation results")
def evaluate(model, data, metrics, save_results) -> None:
    """Evaluate model performance (stub)."""
    console.print("[bold cyan]MRI-KAN Evaluation[/bold cyan]")
    console.print("Evaluation routine not implemented in this demo.")


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
