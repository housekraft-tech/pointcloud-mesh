"""Subset existing CAD faces to a conservative raw-LAS distance envelope.

No refitting, hole filling, extrusion or use of Poisson as evidence. A square
is retained only if its centre's nearest-return distance plus its half-diagonal
is below the cutoff. Since nearest-set distance is 1-Lipschitz, the entire
square is inside that envelope. Ambiguous smallest cells are dropped.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely import box, contains_xy, union_all
from shapely.geometry import Polygon

from audit_soulace_overlap import ROOT, STEMS, raw_points, say, uniform_surface, stats


def overlap_stats(distance):
    result = stats(distance)
    if result is not None:
        for threshold in [20, 50]:
            result[f"within_{threshold}mm_pct"] = round(float((distance <= threshold/1000).mean()*100),2)
        result["beyond_50mm_pct"] = round(float((distance > .05).mean()*100),2)
    return result


def polygon_parts(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == "Polygon":
        return [shape]
    return [p for g in getattr(shape, "geoms", []) for p in polygon_parts(g)]


def plane_patches(mesh):
    """Group only truly coplanar triangles, retaining the original support."""
    groups = defaultdict(list)
    for i, (normal, centre) in enumerate(zip(mesh.face_normals, mesh.triangles_center)):
        if mesh.area_faces[i] < 1e-10:
            continue
        n = normal.copy()
        if n[np.argmax(np.abs(n))] < 0:
            n = -n
        key = tuple(np.round(n, 8)) + (round(float(n @ centre), 7),)
        groups[key].append(i)
    for ids in groups.values():
        if mesh.area_faces[ids].sum() < .02:
            continue
        n = mesh.face_normals[ids[0]]
        seed = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n, seed); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        origin = mesh.triangles[ids[0], 0].copy()
        triangles = mesh.triangles[ids]
        # Do not merge offset parallel surfaces despite rounded grouping keys.
        if np.max(np.abs((triangles - origin) @ n)) > 1e-7:
            raise ValueError("Non-coplanar triangles were grouped")
        projected = np.stack(((triangles-origin) @ u, (triangles-origin) @ v), axis=-1)
        poly = union_all([Polygon(t) for t in projected]).buffer(0)
        yield poly, origin, u, v


def mask_rectangles(mask, x0, y0, step):
    """Exact union of grid cells via vertically merged row runs, not contours."""
    previous = {}
    rectangles = []
    for row in range(mask.shape[0] + 1):
        if row < mask.shape[0]:
            changes = np.flatnonzero(np.diff(np.r_[False, mask[row], False]))
            runs = {(int(a), int(b)) for a, b in changes.reshape(-1, 2)}
        else:
            runs = set()
        for run in previous.keys() - runs:
            a, b = run
            rectangles.append((x0+a*step, y0+previous[run]*step, x0+b*step, y0+row*step))
        previous = {run: previous.get(run, row) for run in runs}
    if not rectangles:
        return Polygon()
    r = np.asarray(rectangles)
    return union_all(box(r[:, 0], r[:, 1], r[:, 2], r[:, 3]))


def supported_polygon(poly, origin, u, v, tree, cutoff=.01,
                      coarse=.04, depth=4, min_area=.02):
    step = coarse / (2 ** depth)
    xmin, ymin, xmax, ymax = poly.bounds
    x0 = math.floor(xmin / coarse) * coarse
    y0 = math.floor(ymin / coarse) * coarse
    nx = max(1, math.ceil((xmax-x0)/coarse))
    ny = max(1, math.ceil((ymax-y0)/coarse))
    yy, xx = np.indices((ny, nx))
    px = x0 + (xx.ravel()+.5)*coarse
    py = y0 + (yy.ravel()+.5)*coarse
    within = contains_xy(poly.buffer(coarse/math.sqrt(2)+1e-8), px, py)
    iy, ix = yy.ravel()[within], xx.ravel()[within]
    mask = np.zeros((ny*2**depth, nx*2**depth), bool)
    queries = 0
    accepted = 0
    bound_max = 0.0
    for level in range(depth+1):
        size = coarse / 2**level
        radius = size / math.sqrt(2)
        if not len(ix):
            break
        xy = np.column_stack((x0+(ix+.5)*size, y0+(iy+.5)*size))
        points = origin + xy[:, 0, None]*u + xy[:, 1, None]*v
        d = np.empty(len(points))
        for start in range(0, len(points), 500_000):
            d[start:start+500_000] = tree.query(points[start:start+500_000], workers=8)[0]
        queries += len(points)
        # 2 micrometres margin accommodates exported coordinate rounding.
        keep = d + radius <= cutoff - .000002
        if keep.any():
            accepted += int(keep.sum())
            bound_max = max(bound_max, float(np.max(d[keep]+radius)))
            factor = 2**(depth-level)
            level_mask = np.zeros((ny*2**level, nx*2**level), bool)
            level_mask[iy[keep], ix[keep]] = True
            mask |= np.repeat(np.repeat(level_mask, factor, 0), factor, 1)
        if level < depth:
            uncertain = (~keep) & (d - radius <= cutoff)
            x, y = ix[uncertain]*2, iy[uncertain]*2
            ix = np.concatenate((x, x+1, x, x+1))
            iy = np.concatenate((y, y, y+1, y+1))
    # Remove disconnected specks, without growing masks or filling holes.
    if mask.any():
        labels, count = ndimage.label(mask)
        sizes = np.bincount(labels.ravel()) * step**2
        valid = sizes >= min_area
        valid[0] = False
        mask = valid[labels]
        del labels
    selected = mask_rectangles(mask, x0, y0, step).intersection(poly)
    selected = union_all([p for p in polygon_parts(selected) if p.area >= min_area])
    return selected, {"queries": queries, "accepted_cells": accepted,
                      "distance_upper_bound_mm": round(bound_max*1000, 6)}


def mesh_from_polygon(poly, origin, u, v):
    meshes = []
    for p in polygon_parts(poly):
        vertices, faces = trimesh.creation.triangulate_polygon(p, engine="earcut")
        xyz = origin + vertices[:, 0, None]*u + vertices[:, 1, None]*v
        mesh = trimesh.Trimesh(xyz, faces, process=False)
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes) if meshes else None


def process_level(level, out, cutoff):
    rng = np.random.default_rng(9210+level)
    source = ROOT / "output_final/soulace_asbuilt_v2"
    metadata = json.loads((source/f"{STEMS[level]}_asbuilt_manifest.json").read_text())
    payload = json.loads((source/f"{STEMS[level]}_asbuilt.build.json").read_text())["parts"]
    say(f"L{level}: loading full LAS reference")
    raw, z_shift = raw_points(ROOT/f"output_final/soulace_L{level}/lidar/L{level}.las",
                              metadata["source_yaw_deg"], metadata["source_floor_z_m"])
    tree = cKDTree(raw, leafsize=32, compact_nodes=False)
    say(f"L{level}: {len(raw):,} returns; filtering {len(payload)} CAD objects")
    kept, records, checks = [], [], []
    for index, item in enumerate(payload):
        mesh = trimesh.Trimesh(np.asarray(item["v"]), np.asarray(item["f"]), process=False)
        patches = []
        query_count = 0
        bound_max = 0
        for poly, origin, u, v in plane_patches(mesh):
            if poly.area < .02:
                continue
            clipped, info = supported_polygon(poly, origin, u, v, tree, cutoff)
            query_count += info["queries"]
            if not clipped.is_empty:
                part = mesh_from_polygon(clipped, origin, u, v)
                if part is not None:
                    patches.append(part)
                    bound_max = max(bound_max, info["distance_upper_bound_mm"])
        result = None
        verification = None
        if patches:
            result = trimesh.util.concatenate(patches)
            result.merge_vertices(digits_vertex=9)
            result.update_faces(result.nondegenerate_faces())
            result.remove_unreferenced_vertices()
            # Re-test exported precision, independently sampled across every
            # retained face, as well as every vertex and edge midpoint.
            result.vertices = np.round(result.vertices, 8)
            sample, _ = uniform_surface(result, np.arange(len(result.faces)),
                                        max(1000, int(result.area*1200)), rng)
            edge_mid = result.vertices[result.edges_unique].mean(axis=1)
            query = np.vstack((sample, result.vertices, edge_mid))
            d = tree.query(query, workers=8)[0]
            if d.max() > cutoff+1e-7:
                raise ValueError(f"L{level}/{item['name']}: export exceeds envelope: {d.max()}")
            checks.append(d)
            verification = {**overlap_stats(d), "maximum_mm": round(float(d.max()*1000), 5)}
            kept.append({"name":item["name"], "kind":item["kind"],
                         "colour":item.get("colour", [191,183,170]),
                         "v": result.vertices.tolist(), "f": result.faces.tolist(),
                         "support_cutoff_mm":cutoff*1000, "open_surface":True})
        record = {"name":item["name"], "kind":item["kind"],
                  "source_area_m2":round(float(mesh.area), 6),
                  "retained_area_m2":round(float(result.area), 6) if result is not None else 0,
                  "retained_triangles":len(result.faces) if result is not None else 0,
                  "distance_queries":query_count, "certified_cell_bound_mm":bound_max,
                  "verification":verification}
        records.append(record)
        if index%5 == 0 or result is not None:
            say(f"L{level} {index+1}/{len(payload)} {item['name']}: {record['retained_area_m2']:.2f}/{mesh.area:.2f} m2 retained")
    all_checks = np.concatenate(checks) if checks else np.array([])
    report = {"level":level, "cutoff_mm":cutoff*1000, "las_points":len(raw),
              "source_yaw_deg":metadata["source_yaw_deg"], "floor_z_m":metadata["source_floor_z_m"],
              "las_z_percentile_0_5_m":z_shift, "source_objects":len(payload),
              "objects_with_retained_surface":len(kept), "objects_entirely_excluded":len(payload)-len(kept),
              "source_surface_area_m2":sum(x["source_area_m2"] for x in records),
              "retained_surface_area_m2":sum(x["retained_area_m2"] for x in records),
              "export_spot_checks":overlap_stats(all_checks),
              "maximum_checked_distance_mm":round(float(all_checks.max()*1000),5) if len(all_checks) else None,
              "objects":records}
    (out/f"Soulace_L{level}_overlap_only.build.json").write_text(json.dumps({"parts":kept}))
    (out/f"Soulace_L{level}_filter_audit.json").write_text(json.dumps(report, indent=2))
    del raw, tree
    gc.collect()
    say(f"L{level}: finished; {len(kept)} objects have retained surfaces")
    return kept, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output_final/soulace_overlap_only_50mm")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--cutoff-mm", type=float, default=50)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    parts, reports = [], []
    storeys = json.loads((ROOT/"soulace_output/storeys.json").read_text())["storeys"]
    for level in map(int, args.levels.split(",")):
        if args.assemble_only:
            payload = json.loads((out/f"Soulace_L{level}_overlap_only.build.json").read_text())["parts"]
            report = json.loads((out/f"Soulace_L{level}_filter_audit.json").read_text())
        else:
            payload, report = process_level(level, out, args.cutoff_mm/1000)
        reports.append(report)
        offset = storeys[level]["floor_z"]-storeys[0]["floor_z"]
        for item in payload:
            item = dict(item)
            vertices = np.asarray(item["v"])
            vertices[:, 2] += offset
            item["v"] = np.round(vertices,8).tolist()
            item["name"] = f"L{level}_{item['name']}"
            item["level"] = level
            parts.append(item)
    (out/"Soulace_overlap_only.build.json").write_text(json.dumps({"parts":parts}))
    audit = {"cutoff_mm":args.cutoff_mm, "levels":reports,
             "method":"Conservative adaptive square inclusion using nearest-return distance plus half-diagonal; original CAD-plane intersection only.",
             "minimum_grid_mm":2.5, "minimum_connected_patch_m2":.02,
             "no_refitting":True, "no_hole_filling":True, "no_added_thickness":True,
             "note":"An incomplete open-surface subset of existing CAD, not an independently validated as-built or proof excluded surfaces are absent."}
    (out/"filter_audit.json").write_text(json.dumps(audit,indent=2))
    say(f"Saved {len(parts)} retained objects to {out}")


if __name__ == "__main__":
    main()
