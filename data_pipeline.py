"""Data pipeline utilities with caching and augmentations."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from data_validation import validate_dataset

try:  # pragma: no cover - optional dependency
    import pydicom
except Exception:  # noqa: S110
    pydicom = None

try:  # pragma: no cover - optional dependency
    import SimpleITK as sitk
except Exception:  # noqa: S110
    sitk = None


class MRIDataset(Dataset):
    """Dataset capable of reading NIfTI and DICOM studies."""

    def __init__(self, root_dir: str, transform=None, cache_size: int = 100, num_workers: int = 4) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_workers = num_workers
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self.target_spacing = (1.0, 1.0, 1.0)

        self.files: list[Path] = []
        for p in sorted(self.root_dir.iterdir()):
            if p.is_file() and p.suffix in {".nii", ".nii.gz"}:
                self.files.append(p)
            elif p.is_dir() and list(p.glob("*.dcm")):
                self.files.append(p)

        failed = validate_dataset(self.files)
        if failed:
            raise ValueError(f"Invalid imaging files detected: {failed}")

        self.compute_statistics()

    @lru_cache(maxsize=32)
    def load_volume(self, path: Path) -> Tuple[np.ndarray, Dict]:
        if path.suffix in {".nii", ".nii.gz"}:
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            meta: Dict = {}
        else:
            data, meta = self.read_dicom_series(path)
        data = (data - self.mean) / self.std
        return data.astype(np.float32), meta

    def read_dicom_series(self, path: Path) -> Tuple[np.ndarray, Dict]:
        if pydicom is None or sitk is None:  # pragma: no cover - dependency check
            raise ImportError("pydicom and SimpleITK are required for DICOM support")
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(series)
        img = reader.Execute()
        img = sitk.DICOMOrient(img, "RAS")
        if self.target_spacing is not None:
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(self.target_spacing)
            size = [
                int(round(sz * spc / tspc))
                for sz, spc, tspc in zip(img.GetSize(), img.GetSpacing(), self.target_spacing)
            ]
            resampler.SetSize(size)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputDirection(img.GetDirection())
            resampler.SetOutputOrigin(img.GetOrigin())
            img = resampler.Execute(img)
        data = sitk.GetArrayFromImage(img).astype(np.float32)
        meta: Dict = {
            "spacing": tuple(reversed(img.GetSpacing())),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
        }
        ds = pydicom.dcmread(series[0])
        meta.update(
            {
                "patient_id": getattr(ds, "PatientID", ""),
                "sex": getattr(ds, "PatientSex", "U"),
                "age": getattr(ds, "PatientAge", "0"),
                "study_date": getattr(ds, "StudyDate", ""),
            }
        )
        return data, meta

    def load_mask(self, path: Path, shape: Tuple[int, ...]) -> np.ndarray:
        if path.suffix in {".nii", ".nii.gz"}:
            mask_path = path.parent / f"{path.stem}_mask.nii.gz"
            if mask_path.exists():
                return nib.load(mask_path).get_fdata().astype(np.float32)
            return np.zeros(shape, dtype=np.float32)
        rt_files = list(path.glob("*RTSTRUCT*.dcm"))
        if rt_files and sitk is not None:
            try:  # pragma: no cover - depends on dataset
                rt = sitk.ReadImage(str(rt_files[0]))
                mask = sitk.GetArrayFromImage(rt).astype(np.float32)
                if mask.shape != shape:
                    mask = np.zeros(shape, dtype=np.float32)
                return mask
            except Exception:  # noqa: S110
                return np.zeros(shape, dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)

    def compute_statistics(self) -> None:
        samples = self.files[: min(100, len(self.files))]
        volumes = []
        for p in samples:
            if p.suffix in {".nii", ".nii.gz"}:
                volumes.append(nib.load(p).get_fdata())
            elif sitk is not None and p.is_dir():
                vol, _ = self.read_dicom_series(p)
                volumes.append(vol)
        self.mean = float(np.mean([v.mean() for v in volumes])) if volumes else 0.0
        self.std = float(np.mean([v.std() for v in volumes])) if volumes else 1.0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        volume, meta = self.load_volume(self.files[idx])
        mask = self.load_mask(self.files[idx], volume.shape)
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
            "meta": meta,
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
