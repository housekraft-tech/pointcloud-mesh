"""Point container and noise model.

PointSet keeps every per-point attribute row-aligned with xyz so that any
subsetting operation anywhere in the pipeline cannot silently desynchronise
intensity or timestamps from geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointSet:
    """A point cloud plus its optional per-point attributes, all row-aligned."""

    xyz: np.ndarray                          # (N, 3) float64, metres
    gps_time: Optional[np.ndarray] = None    # (N,) float64
    intensity: Optional[np.ndarray] = None   # (N,)
    rgb: Optional[np.ndarray] = None         # (N, 3) uint8

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N, 3), got {self.xyz.shape}")
        for name in ("gps_time", "intensity", "rgb"):
            attr = getattr(self, name)
            if attr is not None and len(attr) != len(self.xyz):
                raise ValueError(
                    f"{name} row count {len(attr)} != xyz row count {len(self.xyz)}"
                )

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    def subset(self, mask) -> "PointSet":
        """Return a new PointSet keeping rows selected by a boolean or index mask."""
        mask = np.asarray(mask)
        return PointSet(
            xyz=self.xyz[mask],
            gps_time=None if self.gps_time is None else self.gps_time[mask],
            intensity=None if self.intensity is None else self.intensity[mask],
            rgb=None if self.rgb is None else self.rgb[mask],
        )


def add_gaussian_noise(
    xyz: np.ndarray, sigma_m: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a copy of xyz with isotropic Gaussian noise of the given sigma.

    The generator is passed in rather than seeded internally so that callers
    control reproducibility. Never mutates the input.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if sigma_m < 0:
        raise ValueError(f"sigma_m must be non-negative, got {sigma_m}")
    if sigma_m == 0:
        return xyz.copy()
    return xyz + rng.normal(0.0, sigma_m, size=xyz.shape)
