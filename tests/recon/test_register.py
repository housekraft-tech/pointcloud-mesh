import numpy as np
from scripts.recon.drawing import DrawingModel
from scripts.recon.register import register


def _rect_mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape, np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_register_recovers_scale_and_translation_of_a_footprint():
    # LiDAR footprint: a 120x80 rectangle at (40,30) in a 200x200 frame.
    lfoot = _rect_mask((200, 200), 40, 30, 160, 110)
    lwall = np.zeros((200, 200), np.uint8)
    lwall[30:110, 40] = 1
    lwall[30:110, 159] = 1
    lwall[30, 40:160] = 1
    lwall[109, 40:160] = 1
    # Drawing footprint: same rectangle at half the pixel scale, offset elsewhere.
    dfoot = _rect_mask((300, 300), 10, 10, 70, 50)     # 60x40, scale ~x2 to match
    dwall = np.zeros((300, 300), np.uint8)
    dwall[10:50, 10] = 1
    dwall[10:50, 69] = 1
    dwall[10, 10:70] = 1
    dwall[49, 10:70] = 1
    seeds = [dict(name="Room", cx=40, cy=30)]
    wall_segs = [(np.array([10., 10.]), np.array([10., 50.]))]
    dm = DrawingModel((0, 0), dwall, dfoot, seeds, wall_segs)

    reg = register(dm, lfoot, lidar_occ=lfoot, lidar_wall=lwall)

    assert reg.footprint_iou > 0.9
    # a drawing corner (10,10) should map near the LiDAR rectangle corner (40,30)
    q = reg.transform.apply(np.array([[10., 10.]]))[0]
    assert abs(q[0] - 40) < 6 and abs(q[1] - 30) < 6
