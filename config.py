import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    in_channels: int = 2
    cube_size: int = 8
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    use_flash_attention: bool = True
    use_sparse_attention: bool = True
    attention_window: int = 256
    use_moe: bool = False
    num_experts: int = 4


@dataclass
class TrainingConfig:
    """Training configuration"""

    batch_size: int = 8
    adaptive_batch: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 8
    batch_size_schedule: Optional[List[int]] = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    gradient_accumulation: int = 4
    distributed: bool = False
    num_workers: int = 8


@dataclass
class DataConfig:
    """Data configuration"""

    data_root: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    augmentation: bool = True
    normalize: bool = True
    patch_size: tuple = (128, 128, 128)
    overlap: float = 0.25
    preprocessors: List[str] = field(default_factory=lambda: ["normalize", "resample"])


@dataclass
class LossConfig:
    """Loss configuration"""

    seg_weight: float = 1.0
    cls_weight: float = 1.0
    edge_weight: float = 0.5
    tumor_weight: float = 1.0
    use_dice: bool = True
    use_focal: bool = True
    use_boundary: bool = True
    label_smoothing: float = 0.1


@dataclass
class Config:
    """Main configuration"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    experiment_name: str = "mri_kan_sota"
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
