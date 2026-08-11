"""A synthetic bare-shell room with exactly known dimensions.

Only interior-visible faces are sampled, mirroring what a scanner standing in
the room actually sees. Truth values are the numbers the pipeline must recover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rscene.core.points import add_gaussian_noise
from rscene.core.prim import Box

# Room interior: x in [0, 3.0], y in [0, 2.5], z in [0, 2.75]
CLEAR_X = 3.0
CLEAR_Y = 2.5
HEIGHT = 2.75
EXTRUSION_DEPTH = 0.075
EXTRUSION_WIDTH = 0.35
GROOVE_DEPTH = 0.012
GROOVE_WIDTH = 0.06
BOX_SIZE = 0.08
BOX_DEPTH = 0.045


@dataclass
class GoldenScene:
    points: np.ndarray
    truth: dict[str, float]


def build_golden_room(
    spacing_m: float = 0.008, noise_m: float = 0.001, seed: int = 0
) -> GoldenScene:
    """Build the room and sample its interior surfaces."""
    faces: list[np.ndarray] = []

    # floor and ceiling
    faces.append(Box("floor", (0, 0, 0), (CLEAR_X, CLEAR_Y, 0)).sample_surface(
        spacing_m, faces=("z+",)))
    faces.append(Box("ceiling", (0, 0, HEIGHT), (CLEAR_X, CLEAR_Y, HEIGHT)).sample_surface(
        spacing_m, faces=("z-",)))

    # wall at y = 0, carrying the extrusion (x 1.20 -> 1.55)
    ex0, ex1 = 1.20, 1.20 + EXTRUSION_WIDTH
    faces.append(Box("y0_left", (0, 0, 0), (ex0, 0, HEIGHT)).sample_surface(
        spacing_m, faces=("y+",)))
    faces.append(Box("y0_right", (ex1, 0, 0), (CLEAR_X, 0, HEIGHT)).sample_surface(
        spacing_m, faces=("y+",)))
    faces.append(Box("ext_face", (ex0, EXTRUSION_DEPTH, 0),
                     (ex1, EXTRUSION_DEPTH, HEIGHT)).sample_surface(
        spacing_m, faces=("y+",)))
    faces.append(Box("ext_side_l", (ex0, 0, 0), (ex0, EXTRUSION_DEPTH, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))
    faces.append(Box("ext_side_r", (ex1, 0, 0), (ex1, EXTRUSION_DEPTH, HEIGHT)).sample_surface(
        spacing_m, faces=("x-",)))

    # wall at y = CLEAR_Y, plain
    faces.append(Box("y1", (0, CLEAR_Y, 0), (CLEAR_X, CLEAR_Y, HEIGHT)).sample_surface(
        spacing_m, faces=("y-",)))

    # wall at x = 0, carrying a full-height groove (y 1.00 -> 1.06)
    gr0, gr1 = 1.00, 1.00 + GROOVE_WIDTH
    faces.append(Box("x0_a", (0, 0, 0), (0, gr0, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))
    faces.append(Box("x0_b", (0, gr1, 0), (0, CLEAR_Y, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))
    faces.append(Box("groove_base", (-GROOVE_DEPTH, gr0, 0),
                     (-GROOVE_DEPTH, gr1, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))

    # wall at x = CLEAR_X, carrying a recessed switch box at z 1.20 -> 1.28
    sb_y0, sb_y1 = 1.10, 1.10 + BOX_SIZE
    sb_z0, sb_z1 = 1.20, 1.20 + BOX_SIZE
    # the wall face is sampled as a ring around the recess -- a scanner cannot
    # see wall surface where the box has been cut out of it
    faces.append(Box("x1_below", (CLEAR_X, 0, 0), (CLEAR_X, CLEAR_Y, sb_z0)).sample_surface(
        spacing_m, faces=("x-",)))
    faces.append(Box("x1_above", (CLEAR_X, 0, sb_z1),
                     (CLEAR_X, CLEAR_Y, HEIGHT)).sample_surface(spacing_m, faces=("x-",)))
    faces.append(Box("x1_left", (CLEAR_X, 0, sb_z0),
                     (CLEAR_X, sb_y0, sb_z1)).sample_surface(spacing_m, faces=("x-",)))
    faces.append(Box("x1_right", (CLEAR_X, sb_y1, sb_z0),
                     (CLEAR_X, CLEAR_Y, sb_z1)).sample_surface(spacing_m, faces=("x-",)))
    # the recess base, sampled finely enough to clear the min_patch_points floor
    faces.append(Box("sb_base", (CLEAR_X + BOX_DEPTH, sb_y0, sb_z0),
                     (CLEAR_X + BOX_DEPTH, sb_y1, sb_z1)).sample_surface(
        0.004, faces=("x-",)))

    points = np.concatenate(faces)
    points = add_gaussian_noise(points, noise_m, np.random.default_rng(seed))

    truth = {
        "clear_span_x_m": CLEAR_X,
        "clear_span_y_m": CLEAR_Y,
        "ceiling_height_m": HEIGHT,
        "extrusion_depth_m": EXTRUSION_DEPTH,
        "extrusion_width_m": EXTRUSION_WIDTH,
        "groove_depth_m": GROOVE_DEPTH,
        "switch_box_depth_m": BOX_DEPTH,
    }
    return GoldenScene(points=points, truth=truth)
