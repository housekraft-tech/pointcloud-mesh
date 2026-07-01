import json
from scripts.floorplan_schema import Wall, Opening, wall_to_dict, new_wall_id


def test_new_wall_id_format():
    assert new_wall_id(3) == "wall_003"


def test_wall_to_dict_round_trips_through_json():
    opening = Opening(
        opening_id="wall_000_op_00", wall_id="wall_000", type="door",
        u_min_m=2.0, u_max_m=2.9, sill_m=0.0, height_m=2.1, width_m=0.9,
        edge_method="density_half_max", both_faces_confirmed=True,
    )
    wall = Wall(
        wall_id="wall_000", p0=(0.0, 0.0), p1=(5.0, 0.0), length_m=5.0,
        thickness_m=0.1, thickness_source="measured",
        plane_front=[1.0, 0.0, 0.0, -2.95], plane_back=[1.0, 0.0, 0.0, -3.05],
        origin_xyz=(2.95, 0.0, 0.0), u_axis=(0.0, 1.0, 0.0), v_axis=(0.0, 0.0, 1.0),
        floor_z_m=0.0, ceiling_z_m=2.7, region_band_m=0.025, region_corner_margin_m=0.5,
        openings=[opening],
    )
    d = wall_to_dict(wall)
    text = json.dumps(d)  # must not raise
    loaded = json.loads(text)
    assert loaded["wall_id"] == "wall_000"
    assert loaded["openings"][0]["type"] == "door"
    assert loaded["grooves"] == []
