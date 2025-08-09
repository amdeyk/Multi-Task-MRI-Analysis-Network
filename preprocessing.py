"""Standardized preprocessing pipelines for medical images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import SimpleITK as sitk
except Exception:  # noqa: S110
    sitk = None

try:  # pragma: no cover - optional dependency
    from monai.transforms import Resize, Orientation
except Exception:  # noqa: S110
    Resize = Orientation = None


@dataclass
class PreprocessingPipeline:
    """Pipeline handling spacing, orientation and intensity normalisation."""

    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def standardize_orientation(self, image: "sitk.Image") -> "sitk.Image":
        if sitk is None:
            return image
        return sitk.DICOMOrient(image, "RAS")

    def resample(self, image: "sitk.Image") -> "sitk.Image":
        if sitk is None:
            return image
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        size = [
            int(round(sz * spc / tspc))
            for sz, spc, tspc in zip(image.GetSize(), image.GetSpacing(), self.target_spacing)
        ]
        resampler.SetSize(size)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        return resampler.Execute(image)

    def reduce_noise(self, image: "sitk.Image") -> "sitk.Image":
        if sitk is None:
            return image
        filt = sitk.CurvatureFlowImageFilter()
        filt.SetTimeStep(0.125)
        filt.SetNumberOfIterations(5)
        return filt.Execute(image)

    def normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        mean = float(volume.mean())
        std = float(volume.std()) + 1e-8
        return (volume - mean) / std

    def preprocess_image(self, image: "sitk.Image") -> np.ndarray:
        image = self.standardize_orientation(image)
        image = self.resample(image)
        image = self.reduce_noise(image)
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        return self.normalize_intensity(array)

    def preprocess_array(self, array: np.ndarray, metadata: Dict) -> np.ndarray:
        if sitk is None:
            return self.normalize_intensity(array)
        image = sitk.GetImageFromArray(array)
        spacing = metadata.get("spacing")
        if spacing:
            image.SetSpacing(tuple(reversed(spacing)))
        return self.preprocess_image(image)

