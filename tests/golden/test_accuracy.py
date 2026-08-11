"""Accuracy assertions against exactly known ground truth.

Plan 1 asserts what PATCHES can prove. Part-level dimensions (wall thickness,
opening widths) arrive with Plan 2.
"""
import numpy as np
import pytest

from rscene.config import merged_config
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches, unassigned_count

from .apartment import build_golden_room

TOL = 0.003     # 3 mm, matching tau_fit


@pytest.fixture(scope="module")
def extracted():
    scene = build_golden_room()
    normals, curvature = estimate_normals(scene.points, k=24)
    patches, labels = extract_patches(scene.points, normals, curvature, merged_config())
    return scene, patches, labels


def _offsets_along(patches, axis):
    """Sorted plane offsets of patches whose normal points along `axis`."""
    return sorted(abs(p.d) for p in patches if abs(p.normal[axis]) > 0.99)


@pytest.mark.slow
def test_ceiling_height_is_recovered(extracted):
    scene, patches, _ = extracted
    frame = estimate_frame(patches, merged_config())
    assert abs((frame.ceiling_z - frame.floor_z) - scene.truth["ceiling_height_m"]) < TOL


@pytest.mark.slow
def test_extrusion_depth_is_recovered(extracted):
    scene, patches, _ = extracted
    offsets = _offsets_along(patches, axis=1)
    gaps = [round(b - a, 4) for a, b in zip(offsets, offsets[1:])]
    assert any(abs(g - scene.truth["extrusion_depth_m"]) < TOL for g in gaps), \
        f"no 75 mm step among y-facing offsets {offsets}"


@pytest.mark.slow
def test_groove_depth_is_recovered(extracted):
    scene, patches, _ = extracted
    offsets = _offsets_along(patches, axis=0)
    gaps = [round(b - a, 4) for a, b in zip(offsets, offsets[1:])]
    assert any(abs(g - scene.truth["groove_depth_m"]) < TOL for g in gaps), \
        f"no 12 mm groove among x-facing offsets {offsets}"


@pytest.mark.slow
def test_switch_box_is_found_as_its_own_patch(extracted):
    scene, patches, _ = extracted
    offsets = _offsets_along(patches, axis=0)
    gaps = [round(b - a, 4) for a, b in zip(offsets, offsets[1:])]
    assert any(abs(g - scene.truth["switch_box_depth_m"]) < TOL for g in gaps), \
        f"no 45 mm switch box among x-facing offsets {offsets}"


@pytest.mark.slow
def test_almost_every_point_is_explained(extracted):
    scene, _, labels = extracted
    unassigned = unassigned_count(labels)
    assert unassigned / len(scene.points) < 0.05, \
        f"{unassigned} of {len(scene.points)} points joined no patch"
