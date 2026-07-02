"""Tests for scripts/recon/floorplan2d.py: arrangement -> rooms -> DXF/SVG.

Synthetic Wall/Opening fixtures here mirror tests/fixtures.py's modular_house()
KNOWN ground truth (4 exterior walls forming a 6x5m rectangle at centerlines
(0,0)-(6,0)-(6,5)-(0,5)-(0,0), thickness 0.2m) -- Tasks 11/12 (which would
produce real Wall objects from the modular_house() point cloud) are being
built concurrently and aren't available here, so these tests exercise
floorplan2d.py's functions in isolation against known-correct input.

Area convention: build_room_polygons polygonizes wall CENTERLINES (not
interior faces), so the returned polygon's area equals the area enclosed by
the centerline loop -- for this fixture that is exactly 6 * 5 = 30 sq m "by
construction" of the test rectangle, not the smaller interior clear area
(which would subtract each wall's half-thickness). Callers wanting interior
floor area would need to offset the polygon inward by half the surrounding
walls' thickness themselves; that is out of scope here.
"""
import xml.etree.ElementTree as ET

import numpy as np
import ezdxf
import pytest

from scripts.floorplan_schema import Wall, Opening
from scripts.recon.floorplan2d import build_room_polygons, write_dxf, write_svg

SVG_NS = "{http://www.w3.org/2000/svg}"


def _make_wall(wall_id, p0, p1, thickness_m=0.2, openings=None):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    u_axis = tuple((p1 - p0) / length)
    return Wall(
        wall_id=wall_id,
        p0=tuple(p0),
        p1=tuple(p1),
        length_m=length,
        thickness_m=thickness_m,
        thickness_source="assumed",
        plane_front=[0.0, 0.0, 1.0, 0.0],
        plane_back=None,
        origin_xyz=(float(p0[0]), float(p0[1]), 0.0),
        u_axis=u_axis,
        v_axis=(0.0, 0.0, 1.0),
        floor_z_m=0.0,
        ceiling_z_m=2.7,
        region_band_m=0.3,
        region_corner_margin_m=0.1,
        openings=openings or [],
    )


def _make_opening(opening_id, wall_id, u_min, u_max, sill_m=0.0, height_m=2.1, type_="door"):
    return Opening(
        opening_id=opening_id,
        wall_id=wall_id,
        type=type_,
        u_min_m=u_min,
        u_max_m=u_max,
        sill_m=sill_m,
        height_m=height_m,
        width_m=u_max - u_min,
        edge_method="density_half_max",
        both_faces_confirmed=True,
    )


def _rectangle_walls():
    """4 exterior walls forming modular_house()'s 6x5m centerline rectangle."""
    return [
        _make_wall("wall_000", (0, 0), (6, 0)),
        _make_wall("wall_001", (6, 0), (6, 5)),
        _make_wall("wall_002", (6, 5), (0, 5)),
        _make_wall("wall_003", (0, 5), (0, 0)),
    ]


def _rectangle_openings():
    return [
        _make_opening("opening_000", "wall_000", 2.0, 2.9, sill_m=0.0, height_m=2.1, type_="door"),
        _make_opening("opening_001", "wall_002", 4.0, 5.2, sill_m=0.9, height_m=1.2, type_="window"),
    ]


# ---------- build_room_polygons ----------

def test_build_room_polygons_closed_rectangle_returns_one_polygon_with_correct_area():
    walls = _rectangle_walls()
    polygons = build_room_polygons(walls)

    assert len(polygons) == 1
    expected_area = 6.0 * 5.0
    area = polygons[0].area
    assert abs(area - expected_area) / expected_area < 0.02


def test_build_room_polygons_open_loop_returns_no_polygons():
    # Only 3 of the 4 sides -- the boundary never closes, so no room should
    # be hallucinated from it.
    walls = _rectangle_walls()[:3]
    polygons = build_room_polygons(walls)
    assert polygons == []


def test_build_room_polygons_empty_input_returns_empty_list():
    assert build_room_polygons([]) == []


# ---------- write_dxf ----------

def test_write_dxf_has_expected_layers_and_entities(tmp_path):
    walls = _rectangle_walls()
    openings = _rectangle_openings()
    rooms = build_room_polygons(walls)
    out_path = tmp_path / "plan.dxf"

    result = write_dxf(walls, openings, rooms, str(out_path))

    assert out_path.exists()
    assert result == str(out_path)

    doc = ezdxf.readfile(str(out_path))
    layer_names = {layer.dxf.name for layer in doc.layers}
    for expected in ("WALLS", "OPENINGS", "ROOMS", "DIMS"):
        assert expected in layer_names

    msp = doc.modelspace()
    counts = {name: 0 for name in ("WALLS", "OPENINGS", "ROOMS", "DIMS")}
    for entity in msp:
        layer = entity.dxf.layer
        if layer in counts:
            counts[layer] += 1

    for name, count in counts.items():
        assert count >= 1, f"layer {name} has no entities"

    # Exactly one dim/length annotation per wall, one wall outline per wall,
    # one opening marker per opening.
    assert counts["WALLS"] >= len(walls)
    assert counts["DIMS"] >= len(walls)
    assert counts["OPENINGS"] >= len(openings)
    assert counts["ROOMS"] >= len(rooms)


def test_write_dxf_opening_marker_at_correct_u_position(tmp_path):
    walls = _rectangle_walls()
    openings = [_make_opening("opening_000", "wall_000", 2.0, 2.9)]
    rooms = build_room_polygons(walls)
    out_path = tmp_path / "plan_opening.dxf"

    write_dxf(walls, openings, rooms, str(out_path))

    doc = ezdxf.readfile(str(out_path))
    msp = doc.modelspace()
    opening_entities = [e for e in msp if e.dxf.layer == "OPENINGS"]
    assert len(opening_entities) >= 1

    # wall_000 runs (0,0)->(6,0) along +x, so u in [2.0, 2.9] maps to world
    # x in [2.0, 2.9], y ~ 0 (+- half wall thickness for the marker rect).
    entity = opening_entities[0]
    xs = [pt[0] for pt in entity.get_points()]
    ys = [pt[1] for pt in entity.get_points()]
    assert min(xs) == pytest.approx(2.0, abs=1e-6)
    assert max(xs) == pytest.approx(2.9, abs=1e-6)
    assert max(abs(y) for y in ys) <= 0.2  # within wall thickness of centerline


# ---------- write_svg ----------

def test_write_svg_creates_file_with_expected_elements(tmp_path):
    walls = _rectangle_walls()
    openings = _rectangle_openings()
    rooms = build_room_polygons(walls)
    out_path = tmp_path / "plan.svg"

    result = write_svg(walls, openings, rooms, str(out_path))

    assert out_path.exists()
    assert result == str(out_path)

    tree = ET.parse(str(out_path))
    root = tree.getroot()
    polygons = root.findall(f".//{SVG_NS}polygon")
    # 4 wall rectangles + 2 opening markers + 1 room outline.
    assert len(polygons) == len(walls) + len(openings) + len(rooms)


def test_write_svg_empty_walls_still_writes_a_file(tmp_path):
    out_path = tmp_path / "empty.svg"
    write_svg([], [], [], str(out_path))
    assert out_path.exists()
    # Should still be parseable XML, just with no wall/opening/room polygons.
    tree = ET.parse(str(out_path))
    root = tree.getroot()
    assert root.tag == f"{SVG_NS}svg"
