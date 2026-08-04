import numpy as np
from scripts.recon.drawing import DrawingModel
from scripts.recon.register import Transform, Registration
from scripts.recon.rooms import segment_rooms


def test_segment_splits_two_rooms_by_a_dividing_wall():
    free = np.ones((100, 200), np.uint8)
    wall = np.zeros((100, 200), np.uint8)
    wall[:, 99:101] = 1                 # vertical divider at x=100
    free[wall > 0] = 0
    # identity transform: drawing px == raster px
    T = Transform(1.0, 0.0, False, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0))
    reg = Registration(T, footprint_iou=1.0, wall_match_frac=1.0, n_snapped=0, snapped_walls=[])
    seeds = [dict(name="Left", cx=50, cy=50), dict(name="Right", cx=150, cy=50)]
    dm = DrawingModel((0, 0), wall, np.ones((100, 200), np.uint8), seeds, [])

    rooms = segment_rooms(dm, reg, free, wall, cell_m=0.02, xmin=0.0, ymax=2.0)

    assert {r.name for r in rooms} == {"Left", "Right"}
    left = next(r for r in rooms if r.name == "Left")
    right = next(r for r in rooms if r.name == "Right")
    # each room stays on its own side of the divider
    assert left.mask[:, :99].sum() > 0 and left.mask[:, 101:].sum() == 0
    assert right.mask[:, 101:].sum() > 0 and right.mask[:, :99].sum() == 0
    assert left.area_m2 > 0 and right.area_m2 > 0
