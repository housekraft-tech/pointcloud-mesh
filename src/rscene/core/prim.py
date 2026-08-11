"""Rectilinear box primitive and its surface sampler.

Boxes are the vocabulary of a bare-shell concrete building: formwork produces
flat faces meeting at sharp edges. Synthetic scenes are built from boxes so
tests can assert recovered dimensions against exact ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

FACE_KEYS = ("x-", "x+", "y-", "y+", "z-", "z+")

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectilinear box, named for traceability into tests."""

    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def sample_surface(
        self, spacing_m: float, faces: Sequence[str] = FACE_KEYS
    ) -> np.ndarray:
        """Sample a regular grid on the requested faces.

        Returns (N, 3) float64. Grid spacing is adjusted per face so samples
        land exactly on the face boundary, which keeps extents exact for the
        dimension assertions in the golden tests. Deterministic: no RNG here,
        noise is added separately by points.add_gaussian_noise.
        """
        if spacing_m <= 0:
            raise ValueError(f"spacing_m must be positive, got {spacing_m}")

        lo = np.asarray(self.lo, dtype=np.float64)
        hi = np.asarray(self.hi, dtype=np.float64)
        if np.any(hi < lo):
            raise ValueError(f"box {self.name!r} has hi < lo: {self.lo} {self.hi}")

        chunks = []
        for key in faces:
            if key not in FACE_KEYS:
                raise ValueError(f"unknown face key {key!r}, expected one of {FACE_KEYS}")
            axis = _AXIS[key[0]]
            at_hi = key[1] == "+"
            u_ax, v_ax = [a for a in (0, 1, 2) if a != axis]

            nu = max(2, int(round((hi[u_ax] - lo[u_ax]) / spacing_m)) + 1)
            nv = max(2, int(round((hi[v_ax] - lo[v_ax]) / spacing_m)) + 1)
            us = np.linspace(lo[u_ax], hi[u_ax], nu)
            vs = np.linspace(lo[v_ax], hi[v_ax], nv)
            uu, vv = np.meshgrid(us, vs, indexing="ij")

            pts = np.empty((uu.size, 3), dtype=np.float64)
            pts[:, axis] = hi[axis] if at_hi else lo[axis]
            pts[:, u_ax] = uu.ravel()
            pts[:, v_ax] = vv.ravel()
            chunks.append(pts)

        if not chunks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(chunks, axis=0)
