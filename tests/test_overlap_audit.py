"""Numerical guards for the read-only scan/model overlay audit."""
import importlib.util
from pathlib import Path

import numpy as np
import trimesh

PATH = Path(__file__).resolve().parents[1] / "scripts/export/audit_soulace_overlap.py"
SPEC = importlib.util.spec_from_file_location("overlap_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_cube_section_length_and_height():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    lines = audit.section_segments(mesh.vertices, mesh.faces, .123)
    assert np.allclose(lines[:, :, 2], .123)
    assert np.isclose(np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1).sum(), 4.0)


def test_section_outside_mesh_is_empty():
    mesh = trimesh.creation.box()
    assert audit.section_segments(mesh.vertices, mesh.faces, 2.0).shape == (0, 2, 3)


def test_section_samples_do_not_overweight_tiny_triangles():
    lines = np.array([[[0., 0., 0.], [.001, 0., 0.]], [[.001, 0., 0.], [1., 0., 0.]]])
    q = audit.line_samples(lines, .1)
    assert q.shape == (10, 3)
    assert np.allclose(q[:, 0], np.arange(10) * .1 + .05)


def test_yaw_rotation_matches_declared_frame():
    p = np.array([[1., 0., 4.]])
    audit.rotate_xy(p, -90)
    assert np.allclose(p, [[0, 1, 4]])


def test_distance_buckets_are_millimetres():
    value = audit.stats([0.005, .015, .05, .2])
    assert value["within_10mm_pct"] == 25
    assert value["within_30mm_pct"] == 50
    assert value["beyond_100mm_pct"] == 25
