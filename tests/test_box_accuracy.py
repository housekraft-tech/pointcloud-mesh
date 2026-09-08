import numpy as np

from scripts.export.box_accuracy import (
    distance_stats,
    markdown_report,
    overall_surface_status,
    thickness_evidence,
)


def test_distance_stats_passes_only_when_p95_is_inside_tolerance():
    row = distance_stats(np.array([1.0] * 95 + [9.0] * 5), 10.0)
    assert row["status"] == "pass"
    assert row["p95_mm"] <= 10.0


def test_distance_stats_reviews_a_good_median_with_a_bad_tail():
    row = distance_stats(np.array([2.0] * 80 + [25.0] * 20), 10.0)
    assert row["median_mm"] <= 10.0
    assert row["p95_mm"] > 10.0
    assert row["status"] == "review"


def test_distance_stats_fails_when_typical_surface_misses_tolerance():
    row = distance_stats(np.array([11.0, 12.0, 13.0]), 10.0)
    assert row["status"] == "fail"


def test_overall_status_requires_every_present_kind_to_pass():
    assert overall_surface_status({"wall": {"status": "pass"}, "slab": {"status": "pass"}}) == "pass"
    assert overall_surface_status({"wall": {"status": "pass"}, "slab": {"status": "review"}}) == "review"
    assert overall_surface_status({"wall": {"status": "pass"}, "slab": {"status": "fail"}}) == "fail"
    assert overall_surface_status({}) == "fail"


def test_thickness_evidence_never_calls_an_inferred_wall_measured():
    man = {"parts": [
        {"name": "wall_00", "kind": "wall", "solid_thickness_measured": True},
        {"name": "wall_01", "kind": "wall", "solid_thickness_measured": False},
        {"name": "column_00", "kind": "column"},
    ]}
    evidence = thickness_evidence(man)
    assert evidence["total_walls"] == 2
    assert evidence["measured_walls"] == 1
    assert evidence["inferred_or_unmeasured_names"] == ["wall_01"]


def test_markdown_report_denies_absolute_accuracy_without_controls():
    stats = distance_stats(np.array([1.0, 2.0, 3.0]), 10.0)
    report = {
        "tolerance_mm": 10.0,
        "release_status": "fail",
        "surface_fit_status": "pass",
        "room_accuracy": {
            "status": "not_evaluated",
            "criterion": "every room must pass",
            "reason": "no controls",
        },
        "absolute_accuracy_status": "not_verified_without_passing_independent_room_controls",
        "scan_to_box": {"wall": stats, "slab": stats},
        "box_to_scan_support": stats,
        "wall_thickness_evidence": {
            "total_walls": 1,
            "measured_walls": 1,
            "inferred_or_unmeasured_walls": 0,
            "measured_pct": 100.0,
            "inferred_or_unmeasured_names": [],
        },
        "parts": [{
            "name": "wall_00", "kind": "wall", "thickness_evidence": "measured",
            **stats,
        }],
    }
    text = markdown_report(report)
    assert "NOT YET VERIFIED" in text
    assert "independent site control" in text
    assert "wall_00" in text
