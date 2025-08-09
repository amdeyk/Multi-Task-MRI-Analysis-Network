"""Advanced data pipeline supporting medical imaging formats."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from data_validation import validate_dataset, validate_metadata
from metadata_manager import MetadataManager
from preprocessing import PreprocessingPipeline

try:  # pragma: no cover - optional dependency
    import pydicom
except Exception:  # noqa: S110
    pydicom = None

try:  # pragma: no cover - optional dependency
    import SimpleITK as sitk
except Exception:  # noqa: S110
    sitk = None

StudyEntry = List[Union[Path, Tuple[Path, str]]]


class MRIDataset(Dataset):
    """Dataset capable of reading NIfTI and DICOM studies with multiple sequences."""

    def __init__(self, root_dir: str, transform=None, cache_size: int = 100, num_workers: int = 4) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_workers = num_workers
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size

        self.preproc = PreprocessingPipeline()
        self.meta_mgr = MetadataManager()

        self.studies: List[StudyEntry] = []
        for study in sorted(self.root_dir.iterdir()):
            sequences: StudyEntry = []
            if study.is_dir():
                for item in sorted(study.iterdir()):
                    if item.is_file() and item.suffix in {".nii", ".nii.gz"}:
                        sequences.append(item)
                    elif item.is_dir() and list(item.glob("*.dcm")):
                        sequences.extend(self._group_dicom_series(item))
                if not sequences and list(study.glob("*.dcm")):
                    sequences.extend(self._group_dicom_series(study))
            elif study.is_file() and study.suffix in {".nii", ".nii.gz"}:
                sequences.append(study)
            elif study.is_dir() and list(study.glob("*.dcm")):
                sequences.extend(self._group_dicom_series(study))
            if sequences:
                self.studies.append(sequences)

        all_files = [p if isinstance(p, Path) else p[0] for seqs in self.studies for p in seqs]
        failed = validate_dataset(all_files)
        if failed:
            raise ValueError(f"Invalid imaging files detected: {failed}")

        self.mean = 0.0
        self.std = 1.0

    def _group_dicom_series(self, path: Path) -> List[Tuple[Path, str]]:
        if sitk is None:
            return [(path, "")]
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(str(path))
        if not ids:
            return [(path, "")]
        return [(path, sid) for sid in ids]

    @lru_cache(maxsize=32)
    def load_volume(self, item: Union[Path, Tuple[Path, str]]) -> Tuple[np.ndarray, Dict]:
        if isinstance(item, tuple):
            return self.read_dicom_series(item[0], item[1])
        if item.suffix in {".nii", ".nii.gz"}:
            img = nib.load(item)
            data = img.get_fdata().astype(np.float32)
            meta = self.meta_mgr.extract(item)
            data = self.preproc.preprocess_array(data, meta)
            meta = self.meta_mgr.standardise(meta)
            meta = self.meta_mgr.anonymise(meta)
            return data, meta
        return self.read_dicom_series(item, None)

    def read_dicom_series(self, path: Path, series_id: str | None) -> Tuple[np.ndarray, Dict]:
        if pydicom is None or sitk is None:  # pragma: no cover
            raise ImportError("pydicom and SimpleITK are required for DICOM support")
        reader = sitk.ImageSeriesReader()
        if series_id:
            files = reader.GetGDCMSeriesFileNames(str(path), series_id)
        else:
            files = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(files)
        img = reader.Execute()
        array = self.preproc.preprocess_image(img)
        meta = self.meta_mgr.extract(Path(files[0]))
        meta.update(
            {
                "spacing": tuple(reversed(img.GetSpacing())),
                "origin": img.GetOrigin(),
                "direction": img.GetDirection(),
            }
        )
        meta = self.meta_mgr.standardise(meta)
        meta = self.meta_mgr.anonymise(meta)
        sr_files = list(path.glob("*SR*.dcm"))
        if sr_files:
            meta["structured_report"] = self.meta_mgr.load_structured_report(sr_files[0])
        return array, meta

    def load_mask(self, item: Union[Path, Tuple[Path, str]], shape: Tuple[int, ...]) -> np.ndarray:
        path = item[0] if isinstance(item, tuple) else item
        if path.suffix in {".nii", ".nii.gz"}:
            mask_path = path.parent / f"{path.stem}_mask.nii.gz"
            if mask_path.exists():
                return nib.load(mask_path).get_fdata().astype(np.float32)
            return np.zeros(shape, dtype=np.float32)
        rt_files = list(path.glob("*RTSTRUCT*.dcm"))
        if rt_files and sitk is not None:
            try:  # pragma: no cover
                rt = sitk.ReadImage(str(rt_files[0]))
                mask = sitk.GetArrayFromImage(rt).astype(np.float32)
                if mask.shape != shape:
                    mask = np.zeros(shape, dtype=np.float32)
                return mask
            except Exception:  # noqa: S110
                return np.zeros(shape, dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequences = self.studies[idx]
        volumes: List[np.ndarray] = []
        meta: Dict = {}
        for seq in sequences:
            vol, m = self.load_volume(seq)
            if not validate_metadata(m):
                raise ValueError("Invalid metadata detected")
            volumes.append(vol)
            meta = m
        volume = np.stack(volumes)
        mask = self.load_mask(sequences[0], volumes[0].shape)
        if self.transform:
            augmented = self.transform(image=volume, mask=mask)
            volume = augmented["image"]
            mask = augmented["mask"]
        meta["num_sequences"] = len(volumes)
        if volume.ndim == 4:
            meta["temporal_frames"] = volume.shape[1]
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
