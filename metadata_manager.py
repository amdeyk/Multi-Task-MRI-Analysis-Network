"""Medical imaging metadata extraction and standardisation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

try:  # pragma: no cover - optional dependency
    import pydicom
except Exception:  # noqa: S110
    pydicom = None

try:  # pragma: no cover - optional dependency
    import nibabel as nib
except Exception:  # noqa: S110
    nib = None


@dataclass
class MetadataManager:
    """Handle extraction, validation and anonymisation of metadata."""

    required_fields: Iterable[str] = field(default_factory=lambda: ("modality",))

    def extract(self, path: Path) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if path.is_file() and path.suffix in {".nii", ".nii.gz"} and nib is not None:
            img = nib.load(str(path))
            hdr = img.header
            meta.update(
                {
                    "shape": img.shape,
                    "spacing": hdr.get_zooms(),
                    "affine": img.affine.tolist(),
                    "modality": "MR",
                }
            )
        elif path.is_file() and path.suffix == ".dcm" and pydicom is not None:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            meta.update(self._extract_from_dataset(ds))
        elif path.is_dir() and pydicom is not None:
            sample = next(iter(path.glob("*.dcm")))
            ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
            meta.update(self._extract_from_dataset(ds))
        return meta

    def _extract_from_dataset(self, ds: "pydicom.Dataset") -> Dict[str, Any]:
        return {
            "patient_id": getattr(ds, "PatientID", ""),
            "patient_sex": getattr(ds, "PatientSex", ""),
            "patient_age": getattr(ds, "PatientAge", ""),
            "study_date": getattr(ds, "StudyDate", ""),
            "modality": getattr(ds, "Modality", ""),
            "manufacturer": getattr(ds, "Manufacturer", ""),
            "sequence_name": getattr(ds, "SequenceName", ""),
        }

    def standardise(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            "PatientID": "patient_id",
            "PatientSex": "patient_sex",
            "PatientAge": "patient_age",
            "StudyDate": "study_date",
        }
        for old, new in mapping.items():
            if old in meta and new not in meta:
                meta[new] = meta.pop(old)
        return meta

    def anonymise(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["patient_id", "patient_sex", "patient_age"]:
            if key in meta:
                meta[key] = "anon"
        return meta

    def validate(self, meta: Dict[str, Any]) -> bool:
        return all(field in meta and meta[field] for field in self.required_fields)

    def load_structured_report(self, path: Path) -> Dict[str, Any]:
        if pydicom is None:
            return {}
        try:  # pragma: no cover - dataset dependent
            ds = pydicom.dcmread(str(path))
            return {"series_description": getattr(ds, "SeriesDescription", "")}
        except Exception:  # noqa: S110
            return {}

