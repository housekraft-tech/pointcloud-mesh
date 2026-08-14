"""Command-line entry point.

    rscene patches <scan.las> <out_dir> [--max-points N] [--seed N] [--set KEY=VALUE ...]

Writes scene.json (the source of truth) and report.md (human-readable).
Determinism note: the provenance timestamp is intentionally derived from the
scan's content hash rather than the wall clock, so two runs of the same scan
produce byte-identical output.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __version__
from .config import merged_config
from .core.graph import coplanarity_classes, patch_adjacency
from .core.level import estimate_frame
from .core.normals import estimate_normals
from .core.patches import extract_patches, unassigned_count
from .core.scene import Provenance, Scene, scene_to_json

# Order in which unassigned points are classified. Each check is applied in
# this sequence and a point is placed in the FIRST bucket it matches, so the
# buckets are mutually exclusive by construction and always sum to the
# unassigned total:
#   1. high_curvature             -- curvature too high to have seeded or
#                                     been recruited by any established patch
#   2. plane_exists_but_unassigned -- curvature is fine AND a finalised patch
#                                     plane (agreeing in normal within
#                                     patch_angle_tol_deg, within tau_fit_m)
#                                     sits right there -- a different story
#                                     from a point with no plane at all: the
#                                     plane exists, this point just never got
#                                     recruited into it
#   3. isolated                   -- no agreeing plane nearby AND fewer than
#                                     _MIN_NEIGHBOURS neighbours within
#                                     patch_connect_radius_m (sparse or at
#                                     the edge of the scan)
#   4. no_plane_within_tolerance  -- no agreeing plane nearby and not
#                                     isolated either: genuinely no plane
#                                     within tolerance
_UNASSIGNED_BUCKET_ORDER = (
    "high_curvature",
    "plane_exists_but_unassigned",
    "isolated",
    "no_plane_within_tolerance",
)

# Below this many neighbours within patch_connect_radius_m, a point is
# considered isolated (sparse/edge-of-scan) rather than simply unmatched.
_MIN_NEIGHBOURS = 3

# Chunk size for the no_plane_within_tolerance matmul pass: at n_patches in
# the low hundreds, a (chunk, n_patches) float64 array costs a few tens of
# MB, keeping memory bounded regardless of how many points are unassigned.
_CLASSIFY_CHUNK = 20_000


def _classify_unassigned(
    xyz: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    labels: np.ndarray,
    patches: list,
    config: dict,
) -> dict[str, int]:
    """Bucket every unassigned point into exactly one cause.

    Buckets are mutually exclusive and sum exactly to the unassigned total:
    each point is tested against the checks in _UNASSIGNED_BUCKET_ORDER, in
    that order, and lands in the first one that matches. Fully vectorised
    (chunked matmuls against every patch plane, one batched k-NN query) --
    no per-point Python loop -- so this stays cheap as the unassigned count
    and patch count both grow with scan size.
    """
    counts = {name: 0 for name in _UNASSIGNED_BUCKET_ORDER}
    unassigned_idx = np.nonzero(np.asarray(labels) == -1)[0]
    if len(unassigned_idx) == 0:
        return counts

    max_curvature = float(config["patch_max_curvature"])
    tau = float(config["tau_fit_m"])
    cos_tol = float(np.cos(np.radians(config["patch_angle_tol_deg"])))
    radius = float(config["patch_connect_radius_m"])

    # 1. high_curvature -- already a vectorised comparison.
    curv = curvature[unassigned_idx]
    is_high_curv = curv > max_curvature
    counts["high_curvature"] = int(np.count_nonzero(is_high_curv))

    remaining = unassigned_idx[~is_high_curv]

    # 2. plane_exists_but_unassigned -- for the points that survive gate 1,
    # test every remaining point against every patch plane at once: a plane
    # "agrees" only if its normal is within patch_angle_tol_deg of the
    # point's own normal (the same gate region growth applies), and only
    # agreeing planes are checked for distance. Chunked over points so the
    # (chunk, n_patches) intermediate arrays stay bounded in size.
    # `matched` points DO have a plane within tolerance -- they belong in
    # plane_exists_but_unassigned, not no_plane_within_tolerance.
    matched = np.zeros(len(remaining), dtype=bool)
    if len(patches) and len(remaining):
        patch_normals = np.array([p.normal for p in patches])   # (P, 3)
        patch_ds = np.array([p.d for p in patches])              # (P,)
        for start in range(0, len(remaining), _CLASSIFY_CHUNK):
            chunk = remaining[start:start + _CLASSIFY_CHUNK]
            u_xyz = xyz[chunk]                                    # (C, 3)
            u_normals = normals[chunk]                            # (C, 3)
            agrees = np.abs(u_normals @ patch_normals.T) >= cos_tol      # (C, P)
            dists = np.abs(u_xyz @ patch_normals.T + patch_ds[None, :])  # (C, P)
            matched[start:start + _CLASSIFY_CHUNK] = np.any(agrees & (dists <= tau), axis=1)
    counts["plane_exists_but_unassigned"] = int(np.count_nonzero(matched))

    remaining2 = remaining[~matched]

    # 3. isolated -- one batched k-NN query instead of a per-point radius
    # query. tree.query returns ascending distances per row with the point
    # itself as its own 0-distance nearest neighbour (it's in the tree), so
    # asking for k = _MIN_NEIGHBOURS + 1 neighbours and taking the last
    # column gives the distance to the _MIN_NEIGHBOURS-th *other* point --
    # exactly what the original per-point
    # `len(query_ball_point(..., radius)) - 1 < _MIN_NEIGHBOURS` check
    # tested, just computed for every point in one call.
    if len(remaining2):
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz)
        k_query = min(_MIN_NEIGHBOURS + 1, len(xyz))
        knn_dist, _ = tree.query(xyz[remaining2], k=k_query, workers=-1)
        if k_query == 1:
            knn_dist = knn_dist[:, None]
        kth_dist = knn_dist[:, -1]
        is_isolated = kth_dist > radius
        counts["isolated"] = int(np.count_nonzero(is_isolated))
        # 4. no_plane_within_tolerance -- what's left: curvature is fine, no
        # agreeing plane sits within tau_fit_m, and it isn't isolated
        # either. This is the genuinely-no-plane bucket.
        counts["no_plane_within_tolerance"] = int(len(remaining2) - counts["isolated"])

    return counts


def _report(scene: Scene, n_points: int, unassigned_buckets: dict[str, int]) -> str:
    lines = [
        "# rscene patch extraction report",
        "",
        f"- Scan: `{scene.provenance.scan_path}`",
        f"- SHA256: `{scene.provenance.scan_sha256}`",
        f"- Pipeline version: {scene.provenance.pipeline_version}",
        "",
        "## Points",
        "",
        f"- Loaded: {n_points:,}",
        f"- Unassigned points: {scene.unassigned_points:,} "
        f"({scene.unassigned_points / max(n_points, 1):.1%})",
        "",
        "### Unassigned breakdown",
        "",
        "Each unassigned point is classified into exactly one bucket, checked "
        "in this order: " + " -> ".join(_UNASSIGNED_BUCKET_ORDER) + ". "
        "Buckets are mutually exclusive and sum to the unassigned total.",
        "",
        "| bucket | count | % of unassigned |",
        "|--------|-------|------------------|",
    ]
    total_unassigned = max(scene.unassigned_points, 1)
    for name in _UNASSIGNED_BUCKET_ORDER:
        count = unassigned_buckets.get(name, 0)
        lines.append(f"| {name} | {count:,} | {count / total_unassigned:.1%} |")

    lines += [
        "",
        "## Patches",
        "",
        f"- Extracted: {len(scene.patches)}",
        f"- Coplanarity classes: {len(scene.coplanarity_classes)}",
        f"- Adjacent pairs: {len(scene.adjacency)}",
        "",
        "## Frame (measured, not applied)",
        "",
        f"- Gravity axis: {scene.frame.z_axis}",
        f"- XY rotation: {scene.frame.xy_rotation_deg:.2f} deg",
        f"- Floor Z: {scene.frame.floor_z}",
        f"- Ceiling Z: {scene.frame.ceiling_z}",
        "",
        "## Largest patches",
        "",
        "| id | n_points | p95 residual (mm) | normal |",
        "|----|----------|-------------------|--------|",
    ]
    for p in sorted(scene.patches, key=lambda q: -q.n_points)[:15]:
        normal = ", ".join(f"{v:+.3f}" for v in p.normal)
        lines.append(
            f"| {p.patch_id} | {p.n_points:,} | {p.p95_residual_m * 1000:.1f} | {normal} |"
        )
    return "\n".join(lines) + "\n"


def _parse_set_overrides(raw: list[str] | None) -> dict:
    """Parse repeated --set KEY=VALUE options into a config override dict.

    Values are parsed as JSON scalars when possible (so `64` becomes an int,
    `1.5` a float, `true` a bool) and fall back to the raw string otherwise.
    """
    import json

    overrides: dict = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rscene")
    sub = parser.add_subparsers(dest="command", required=True)

    patches_cmd = sub.add_parser("patches", help="extract planar patches from a scan")
    patches_cmd.add_argument("input", help="path to a LAS/LAZ scan")
    patches_cmd.add_argument("out_dir", help="output directory")
    patches_cmd.add_argument(
        "--max-points", type=int, default=None,
        help="randomly subsample to this many points before extraction. "
             "WARNING: for quick previews/tests only -- see stderr warning.",
    )
    patches_cmd.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (default: 0, or the config default's value); "
             "conflicts with an explicit --set seed=VALUE that disagrees",
    )
    patches_cmd.add_argument(
        "--set", action="append", default=None, metavar="KEY=VALUE",
        help="override a config key, e.g. --set patch_neighbor_k=64; may be repeated",
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    if args.max_points is not None:
        print(
            "warning: --max-points randomly subsamples the point cloud. "
            "Random subsampling destroys the local point density that "
            "region-growing patch extraction depends on and degrades "
            "extraction quality (measured on real data: 13.7% unassigned "
            "at native 6.1 mm spacing vs. 40.9% at 15.2 mm after random "
            "thinning to the same point count). Use --max-points only for "
            "quick previews. For real runs, bound memory by processing a "
            "spatial subregion of the scan instead of random subsampling.",
            file=sys.stderr,
        )

    from .io.las import file_sha256, load_las      # imported late: optional dependency

    try:
        overrides = _parse_set_overrides(args.set)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # --seed has an implicit default (argparse can't tell "not given" from
    # "given as the default"), so only treat it as explicit when the caller
    # actually passed it. An explicit --set seed=VALUE that disagrees with
    # an explicit --seed is an error rather than one silently overwriting
    # the other; if only one is given, it wins outright.
    if args.seed is not None:
        if "seed" in overrides and overrides["seed"] != args.seed:
            print(
                f"error: --seed {args.seed} conflicts with "
                f"--set seed={overrides['seed']!r}; pass only one",
                file=sys.stderr,
            )
            return 2
        overrides.setdefault("seed", args.seed)

    try:
        config = merged_config(overrides)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    points = load_las(args.input, max_points=args.max_points, seed=config["seed"])
    if points.n < config["normal_k"]:
        print(f"error: only {points.n} points, need at least {config['normal_k']}",
              file=sys.stderr)
        return 2

    normals, curvature = estimate_normals(points.xyz, k=config["normal_k"])
    patch_list, labels = extract_patches(points.xyz, normals, curvature, config)
    unassigned_buckets = _classify_unassigned(
        points.xyz, normals, curvature, labels, patch_list, config
    )

    scene = Scene(
        provenance=Provenance(
            scan_path=os.path.basename(args.input),
            scan_sha256=file_sha256(args.input),
            pipeline_version=__version__,
            timestamp=f"content:{file_sha256(args.input)[:16]}",
            config=config,
        ),
        frame=estimate_frame(patch_list, config),
        patches=patch_list,
        coplanarity_classes=coplanarity_classes(patch_list, config),
        adjacency=patch_adjacency(patch_list, points.xyz, config),
        unassigned_points=unassigned_count(labels),
        diagnostics={
            "n_input_points": points.n,
            "p95_residual_m": float(np.percentile(
                [p.p95_residual_m for p in patch_list], 95)) if patch_list else 0.0,
            "unassigned_buckets": unassigned_buckets,
        },
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "scene.json"), "w") as handle:
        handle.write(scene_to_json(scene))
    with open(os.path.join(args.out_dir, "report.md"), "w") as handle:
        handle.write(_report(scene, points.n, unassigned_buckets))

    print(f"{len(patch_list)} patches, {scene.unassigned_points:,} unassigned "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
