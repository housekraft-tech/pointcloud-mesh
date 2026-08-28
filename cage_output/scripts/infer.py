"""Run pretrained CAGE on a single density map, on CPU, on Windows.

The repo needs Linux + CUDA 11.1 + two compiled extensions. Neither is needed
for forward-only inference:
  * diff_ras.polygon.SoftPolygon is a training rasterisation loss -- stubbed.
  * MultiScaleDeformableAttention has a pure-PyTorch reference implementation
    shipped in the same file (ms_deform_attn_core_pytorch) -- patched in.
Everything else is stock torch.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

# The repo hardcodes .cuda() in the denoising path (models/dn_components.py).
# Forward-only on CPU, make it a no-op.
torch.Tensor.cuda = lambda self, *a, **k: self          # type: ignore[assignment]
torch.nn.Module.cuda = lambda self, *a, **k: self       # type: ignore[assignment]

CAGE = Path(r"C:/Users/PC/AppData/Local/Temp/claude/C--Users-PC-Documents-pointcloud-mesh/faf1a5e6-1047-47fa-b574-2e849516d861/scratchpad/CAGE")
sys.path.insert(0, str(CAGE))

# --- stub the training-only rasterisation loss -----------------------------
_dr = types.ModuleType("diff_ras")
_drp = types.ModuleType("diff_ras.polygon")


class SoftPolygon:                                   # noqa: D401
    def __init__(self, *a, **k):
        pass


_drp.SoftPolygon = SoftPolygon
_dr.polygon = _drp
sys.modules["diff_ras"] = _dr
sys.modules["diff_ras.polygon"] = _drp

# --- stub the compiled CUDA op so the import succeeds ----------------------
_msda = types.ModuleType("MultiScaleDeformableAttention")
sys.modules["MultiScaleDeformableAttention"] = _msda

from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch  # noqa: E402
import models.ops.modules.ms_deform_attn as _mod                                   # noqa: E402


class _CPUDeformAttn:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations,
              attention_weights, im2col_step):
        return ms_deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)


_mod.MSDeformAttnFunction = _CPUDeformAttn

from models import build_model                                                     # noqa: E402
from util.edge_utils import (remove_short_edges, get_corners_from_edges,           # noqa: E402
                             merge_points, refine_rooms, remove_rooms_with_iou)
from shapely.geometry import Polygon                                               # noqa: E402


def make_args(backbone, checkpoint, num_queries=800, num_polys=20):
    return argparse.Namespace(
        batch_size=1, backbone=backbone, lr_backbone=2e-4, dilation=False,
        position_embedding='sine', position_embedding_scale=2 * np.pi,
        num_feature_levels=4, enc_layers=6, dec_layers=6, dim_feedforward=1024,
        hidden_dim=256, dropout=0.1, nheads=8, num_queries=num_queries,
        num_polys=num_polys, dec_n_points=4, enc_n_points=4,
        query_pos_type='sine', with_poly_refine=True, masked_attn=False,
        semantic_classes=-1, aux_loss=False, dataset_name='stru3d',
        dataset_root='data/stru3d', eval_set='test', device='cpu',
        num_workers=0, seed=42, checkpoint=checkpoint, output_dir='out',
        use_angle_loss=True, plot_pred=True, plot_density=True, plot_gt=False,
        use_checkpoint=True,
    )


def decode(outputs):
    """Port of engine.evaluate_floor's polygon extraction (non-semantic path)."""
    pred_logits = torch.sigmoid(outputs['pred_logits'])
    pred_corners = outputs['pred_coords']
    fg_mask = pred_logits > 0.5

    room_polys, raw_polys = [], []
    for j in range(fg_mask.shape[1]):
        fg = fg_mask[0, j]
        logits_room = pred_logits[0, j][fg].cpu().numpy()
        valid = pred_corners[0, j][fg]
        if len(valid) == 0:
            continue
        corners = (valid * 255).cpu().numpy()
        raw_polys.append(np.around(corners).astype(np.int32))
        corners, logits_room = remove_short_edges(corners, logits_room)
        corners = np.around(corners).astype(np.int32)
        corners = get_corners_from_edges(corners, logits_room, 10)
        corners = np.around(corners).astype(np.int32)
        corners = merge_points(corners, 2)
        if len(corners) >= 4:
            room_polys.append(corners)

    refined = room_polys
    try:
        shp = []
        for arr in room_polys:
            pts = [tuple(p) for p in arr]
            if pts and pts[0] != pts[-1]:
                pts.append(pts[0])
            shp.append(Polygon(pts))
        shp = remove_rooms_with_iou(shp)
        polygon_list, _ = refine_rooms(shp, False)
        refined = [np.array(p.exterior.coords, dtype=np.int32)[:-1]
                   for p in polygon_list]
    except Exception as e:                                    # noqa: BLE001
        print(f"  refine_rooms failed ({e}); using unrefined polygons")

    return room_polys, refined, raw_polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--density', required=True)
    ap.add_argument('--backbone', default='resnet50')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args_cli = ap.parse_args()

    density = np.load(args_cli.density).astype(np.float32)
    print(f"density {density.shape} max={density.max():.3f} "
          f"nonzero={(density>0).sum()}", flush=True)

    args = make_args(args_cli.backbone, args_cli.checkpoint)
    torch.manual_seed(42)
    model = build_model(args, train=False)
    model.eval()

    ckpt = torch.load(args_cli.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected
                  if not (k.endswith('total_params') or k.endswith('total_ops'))]
    print(f"loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected",
          flush=True)
    if missing[:5]:
        print("  missing e.g.", missing[:5])
    if unexpected[:5]:
        print("  unexpected e.g.", unexpected[:5])

    samples = [torch.from_numpy(density)[None]]  # list of (1,H,W)
    with torch.no_grad():
        outputs, _ = model(samples)

    room_polys, refined, raw = decode(outputs)
    print(f"rooms: {len(room_polys)} decoded -> {len(refined)} after refine",
          flush=True)

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "density": args_cli.density,
        "backbone": args_cli.backbone,
        "n_decoded": len(room_polys),
        "n_refined": len(refined),
        "polys_decoded": [p.tolist() for p in room_polys],
        "polys_refined": [p.tolist() for p in refined],
    }, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
