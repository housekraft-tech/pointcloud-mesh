import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/export'))
from render_native_geometry import render


def test_planar_surface_is_not_mistaken_for_depth_edges(tmp_path):
    part = {'v': [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
            'f': [[0, 1, 2], [0, 2, 3]], 'colour': [200, 200, 200]}
    a, b = tmp_path / 'edges.png', tmp_path / 'plain.png'
    render([part], a, '', size=(400, 350), edges=True)
    render([part], b, '', size=(400, 350), edges=False)
    marked, plain = np.asarray(Image.open(a)), np.asarray(Image.open(b))
    surface = (plain[:, :, 0] > 130) & (plain[:, :, 0] < 230)
    assert surface.sum() > 1000
    assert np.mean(np.any(marked[surface] != plain[surface], axis=1)) < .01
