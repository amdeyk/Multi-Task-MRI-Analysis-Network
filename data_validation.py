"""Data validation utilities for medical imaging datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

try:  # pragma: no cover - optional dependency
    import pydicom
except Exception:  # noqa: BLE001
    pydicom = None

from metadata_manager import MetadataManager

_metadata_mgr = MetadataManager()


def validate_dicom(path: Path) -> bool:
    """Validate a DICOM file or directory."""
    if pydicom is None:
        return True
    try:
        if path.is_dir():
            sample = next(iter(path.glob("*.dcm")))
            pydicom.dcmread(str(sample))
        else:
            pydicom.dcmread(str(path))
        return True
    except Exception:  # noqa: BLE001
        return False


def validate_dataset(files: Iterable[Path]) -> List[Path]:
    """Return a list of files that failed validation checks."""
    failed: List[Path] = []
    for p in files:
        if p.suffix in {".nii", ".nii.gz"}:
            try:
                import nibabel as nib  # type: ignore

                nib.load(p)
            except Exception:  # noqa: BLE001
                failed.append(p)
        else:
            if not validate_dicom(p):
                failed.append(p)
    return failed


def compute_statistics(volume: np.ndarray) -> dict:
    """Basic profiling of intensity distribution."""
    return {
        "mean": float(volume.mean()),
        "std": float(volume.std()),
        "min": float(volume.min()),
        "max": float(volume.max()),
    }


def validate_metadata(meta: Dict) -> bool:
    """Validate extracted metadata using standard rules."""
    return _metadata_mgr.validate(meta)
