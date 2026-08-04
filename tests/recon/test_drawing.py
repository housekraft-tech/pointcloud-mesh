import numpy as np
from scripts.recon.drawing import _masks_from_dets


def _det(name, box):
    return {"name": name, "box": box}


def test_masks_from_dets_crops_and_builds_seeds_and_footprint():
    # 400x400 white image, two dark room rectangles side by side.
    bgr = np.full((400, 400, 3), 255, np.uint8)
    for (x0, y0, x1, y1) in [(40, 40, 180, 360), (220, 40, 360, 360)]:
        bgr[y0:y1, x0:x1] = 240
        bgr[y0:y0 + 6, x0:x1] = 20   # thick dark wall strokes on the room edges
        bgr[y1 - 6:y1, x0:x1] = 20
    room_dets = [_det("Bedroom", [40, 40, 180, 360]),
                 _det("Kitchen", [220, 40, 360, 360])]
    wall_dets = [{"name": "wall", "box": [40, 40, 180, 46]}]

    dm = _masks_from_dets(bgr, room_dets, wall_dets)

    assert len(dm.room_seeds) == 2
    names = {s["name"] for s in dm.room_seeds}
    assert names == {"Bedroom", "Kitchen"}
    for s in dm.room_seeds:
        assert 0 <= s["cx"] < dm.wall_mask.shape[1]
        assert 0 <= s["cy"] < dm.wall_mask.shape[0]
    assert dm.footprint_mask.sum() > 0
    assert dm.wall_mask.sum() > 0
    assert len(dm.wall_segs) == 1
