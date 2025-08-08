"""Data pipeline utilities with caching and augmentations."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class MRIDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, cache_size: int = 100, num_workers: int = 4) -> None:
        self.root_dir = Path(root_dir)
        self.files = list(self.root_dir.glob("*.nii.gz"))
        self.transform = transform
        self.num_workers = num_workers
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self.compute_statistics()

    @lru_cache(maxsize=32)
    def load_volume(self, path: Path) -> np.ndarray:
        img = nib.load(path)
        data = img.get_fdata()
        data = (data - self.mean) / self.std
        return data.astype(np.float32)

    def compute_statistics(self) -> None:
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            volumes = list(ex.map(lambda p: nib.load(p).get_fdata(), self.files[: min(100, len(self.files))]))
        self.mean = np.mean([v.mean() for v in volumes])
        self.std = np.mean([v.std() for v in volumes])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        volume = self.load_volume(self.files[idx])
        mask_path = self.files[idx].parent / f"{self.files[idx].stem}_mask.nii.gz"
        if mask_path.exists():
            mask = nib.load(mask_path).get_fdata()
        else:
            mask = np.zeros_like(volume)
        if self.transform:
            augmented = self.transform(image=volume, mask=mask)
            volume = augmented["image"]
            mask = augmented["mask"]
        return {
            "mri": torch.from_numpy(volume),
            "seg": torch.from_numpy(mask),
            "cls": torch.tensor(0),
            "edge": torch.from_numpy(mask),
            "tumor": torch.from_numpy(mask).long(),
        }


def get_training_augmentation():
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.GaussNoise(var_limit=(0.0, 0.05), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.2),
            ToTensorV2(),
        ]
    )


def get_dataloader(config: Config, split: str = "train") -> DataLoader:
    transform = get_training_augmentation() if split == "train" else ToTensorV2()
    dataset = MRIDataset(
        root_dir=config.data.data_root / split,
        transform=transform,
        cache_size=100,
        num_workers=config.training.num_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == "train"),
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return loader
