"""Accuracy assertions against exactly known ground truth.

Plan 1 asserts what PATCHES can prove. Part-level dimensions (wall thickness,
opening widths) arrive with Plan 2.
"""
from itertools import combinations

import numpy as np
import pytest

from rscene.config import merged_config
from rscene.core.graph import perpendicular_offset
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


def _pairwise_offsets_along(patches, axis):
    """perpendicular_offset for every pair of patches whose normal points
    along `axis` -- never Patch.d, which is origin-referenced and lever-arms
    any normal error by each patch's distance from the world origin (see
    graph.perpendicular_offset's docstring and task-10-report.md)."""
    same_axis = [p for p in patches if abs(p.normal[axis]) > 0.99]
    return [perpendicular_offset(a, b) for a, b in combinations(same_axis, 2)]


@pytest.mark.slow
def test_ceiling_height_is_recovered(extracted):
    scene, patches, _ = extracted
    frame = estimate_frame(patches, merged_config())
    assert abs((frame.ceiling_z - frame.floor_z) - scene.truth["ceiling_height_m"]) < TOL


@pytest.mark.slow
def test_extrusion_depth_is_recovered(extracted):
    scene, patches, _ = extracted
    offsets = _pairwise_offsets_along(patches, axis=1)
    assert any(abs(o - scene.truth["extrusion_depth_m"]) < TOL for o in offsets), \
        f"no 75 mm step among y-facing pairwise offsets {sorted(offsets)}"


@pytest.mark.slow
def test_groove_depth_is_recovered(extracted):
    scene, patches, _ = extracted
    offsets = _pairwise_offsets_along(patches, axis=0)
    assert any(abs(o - scene.truth["groove_depth_m"]) < TOL for o in offsets), \
        f"no 12 mm groove among x-facing pairwise offsets {sorted(offsets)}"


@pytest.mark.slow
def test_switch_box_is_found_as_its_own_patch(extracted):
    scene, patches, _ = extracted
    offsets = _pairwise_offsets_along(patches, axis=0)
    assert any(abs(o - scene.truth["switch_box_depth_m"]) < TOL for o in offsets), \
        f"no 45 mm switch box among x-facing pairwise offsets {sorted(offsets)}"


@pytest.mark.slow
def test_almost_every_point_is_explained(extracted):
    scene, _, labels = extracted
    unassigned = unassigned_count(labels)
    assert unassigned / len(scene.points) < 0.05, \
        f"{unassigned} of {len(scene.points)} points joined no patch"
