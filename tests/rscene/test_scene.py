import json

import numpy as np

from rscene.core.patches import Patch
from rscene.core.scene import (
    Frame, Measurement, Provenance, Scene, scene_from_json, scene_to_json,
)


def _scene():
    return Scene(
        provenance=Provenance(
            scan_path="koushik.las", scan_sha256="abc123",
            pipeline_version="0.1.0", timestamp="2026-08-11T00:00:00Z",
            config={"tau_fit_m": 0.003},
        ),
        frame=Frame(z_axis=[0.0, 0.0, 1.0], xy_rotation_deg=1.4,
                    floor_z=0.0, ceiling_z=2.75),
        patches=[Patch(
            patch_id=0, normal=np.array([1.0, 0.0, 0.0]), d=-2.0,
            point_idx=np.array([0, 1, 2]), n_points=3, p95_residual_m=0.0021,
            centroid=np.array([2.0, 1.0, 1.4]), u_range=(-1.0, 1.0), v_range=(-1.4, 1.4),
        )],
        coplanarity_classes=[[0]],
        adjacency=[],
        unassigned_points=42,
        diagnostics={"p95_residual_m": 0.0021},
    )


def test_measurement_carries_provenance_and_uncertainty():
    m = Measurement(value=0.2031, method="face-to-face raw points",
                    n_points=18422, p95_residual=0.0021)
    assert m.to_dict() == {
        "value": 0.2031, "method": "face-to-face raw points",
        "n_points": 18422, "p95_residual": 0.0021,
    }


def test_scene_round_trips_through_json():
    original = _scene()
    restored = scene_from_json(scene_to_json(original))

    assert restored.provenance.scan_sha256 == "abc123"
    assert restored.frame.ceiling_z == 2.75
    assert restored.unassigned_points == 42
    assert len(restored.patches) == 1
    assert np.allclose(restored.patches[0].normal, [1.0, 0.0, 0.0])
    assert restored.patches[0].d == -2.0
    assert np.array_equal(restored.patches[0].point_idx, [0, 1, 2])


def test_serialisation_is_deterministic():
    assert scene_to_json(_scene()) == scene_to_json(_scene())


def test_json_keys_are_sorted_so_diffs_stay_readable():
    payload = json.loads(scene_to_json(_scene()))
    assert list(payload.keys()) == sorted(payload.keys())


def test_unassigned_points_are_recorded_not_dropped():
    payload = json.loads(scene_to_json(_scene()))
    assert payload["unassigned_points"] == 42
