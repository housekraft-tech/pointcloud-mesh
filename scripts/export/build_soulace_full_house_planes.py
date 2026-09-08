"""Extend the v7 paired-wall and joined-floor treatment to Soulace L0/L1.

The script deliberately reuses wall identities already registered in the
current house model.  It does not discover or insert new wall runs.  Complete
planar solids come from the clean as-built source, while the audit keeps the
measured-vs-modeled thickness distinction explicit.
"""
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
import trimesh


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output_final" / "soulace_wall_floor_junctions_v7"
OUT = ROOT / "output_final" / "soulace_full_house_planes_v8"
NATIVE = "Soulace_full_house_wall_and_floor_planes.skp"
LEVELS = {
    0: ("ground", 0.0),
    1: ("first", 3.2),
}


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [part for part in geometry.geoms if part.geom_type == "Polygon"]


def top_polygon(part):
    vertices = np.asarray(part["v"], dtype=float)
    triangles = vertices[np.asarray(part["f"], dtype=int)]
    top_z = float(vertices[:, 2].max())
    top = triangles[np.all(np.abs(triangles[:, :, 2] - top_z) < 1e-7, axis=1)]
    if not len(top):
        raise ValueError(f"No planar top face in {part['name']}")
    return shapely.union_all(shapely.polygons(top[:, :, :2]))


def remove_tiny_holes(geometry, maximum_area=0.005):
    """Remove only pore-size artifacts; preserve stairs, shafts, and courtyards."""
    clean = []
    removed = []
    for poly in polygons(geometry):
        retained = []
        for ring in poly.interiors:
            hole = Polygon(ring)
            if hole.area <= maximum_area:
                removed.append({"area_m2": float(hole.area), "bounds": list(hole.bounds)})
            else:
                retained.append(ring)
        clean.append(Polygon(poly.exterior, retained))
    return shapely.union_all(clean), removed


def transform_vertices(vertices, level, nominal_z, matrix):
    local = np.asarray(vertices, dtype=float).copy()
    local[:, 2] += nominal_z
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def wall_plane_pair(part, metadata, level, nominal_z, matrix):
    axis = 0 if metadata["axis"] == "x" else 1
    vertices = np.asarray(part["v"], dtype=float)
    cross = [float(vertices[:, axis].min()), float(vertices[:, axis].max())]
    normal = matrix[:3, axis]
    origin = matrix[:3, 3] + matrix[:3, :3] @ np.array([0.0, 0.0, nominal_z])
    verified = part["kind"] == "wall_measured"
    source = (
        "opposing_lidar_wall_faces_measured"
        if verified
        else "registered_wall_pair_modeled_thickness_unverified"
    )
    planes = []
    for side, offset in zip(("A", "B"), cross):
        planes.append(
            {
                "side": side,
                "normal": normal.tolist(),
                "offset_m": float(normal @ origin + offset),
                "source": source,
            }
        )
    return axis, cross, planes, verified


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    additions = OUT / "additions.build.json"
    if additions.exists() or (OUT / NATIVE).exists():
        raise FileExistsError("Use a new Soulace revision; v8 output already exists")

    config = json.loads(
        (ROOT / "output_final/coverage_restored_v1/soulace/restore_manifest.json").read_text()
    )
    source_audit = json.loads((SOURCE / "native_audit.json").read_text())
    base_names = {group["name"] for group in source_audit["groups"]}
    references = set(source_audit["reference_only_groups"])
    parts = []
    audit = {
        "source_native": str((SOURCE / "Soulace_complete_wall_planes_and_floor_junctions.skp").resolve()),
        "existing_groups_preserved": True,
        "new_wall_runs_discovered": 0,
        "levels": {},
        "limitations": (
            "Measured wall thickness is retained only where two-face source evidence was accepted. "
            "Other registered walls use the earlier modeled thickness and are explicitly marked unverified. "
            "Internal CAD joins do not certify site accuracy."
        ),
    }

    for level, (stem, nominal_z) in LEVELS.items():
        matrix = np.asarray(config["source_to_common"][str(level)], dtype=float)
        payload_path = (
            ROOT
            / "output_final/soulace_asbuilt_v2"
            / f"Soulace_L{level}_{stem}_asbuilt.build.json"
        )
        manifest_path = (
            ROOT
            / "output_final/soulace_asbuilt_v2"
            / f"Soulace_L{level}_{stem}_asbuilt_manifest.json"
        )
        source_parts = json.loads(payload_path.read_text())["parts"]
        manifest = json.loads(manifest_path.read_text())
        metadata_by_name = {row["wall"]: row for row in manifest["walls"]}
        wall_parts = [part for part in source_parts if part["kind"].startswith("wall_")]
        floor_parts = [part for part in source_parts if part["kind"] == "floor"]
        expected_walls = {f"L{level}_{part['name']}" for part in wall_parts}
        expected_floors = {f"L{level}_{part['name']}" for part in floor_parts}
        missing_walls = sorted(expected_walls - base_names)
        if missing_walls:
            raise ValueError(f"L{level} has unregistered wall runs: {missing_walls}")
        # A few doorway threshold patches were intentionally absent from the
        # overlap-clipped predecessor.  They are not new rooms or walls: they
        # reconnect already accepted room floor polygons through registered
        # wall openings.  Record them separately instead of hiding the fact.
        restored_thresholds = sorted(expected_floors - base_names)
        references.update(expected_walls)
        references.update(expected_floors & base_names)

        level_rows = []
        for part in wall_parts:
            metadata = metadata_by_name[part["name"]]
            axis, cross, planes, verified = wall_plane_pair(
                part, metadata, level, nominal_z, matrix
            )
            common = transform_vertices(part["v"], level, nominal_z, matrix)
            mesh = trimesh.Trimesh(common, part["f"], process=False)
            if not mesh.is_watertight:
                raise ValueError(f"Wall is not closed: L{level} {part['name']}")
            thickness = cross[1] - cross[0]
            name = f"L{level} paired wall planes {part['name']}"
            new_part = {
                "name": name,
                "kind": "wall_paired_planes",
                "level": level,
                "colour": [205, 198, 181] if verified else [220, 203, 166],
                "v": np.round(common, 8).tolist(),
                "f": part["f"],
                "construction_solid": True,
                "thickness_verified": verified,
                "measured_face_positions_m": cross if verified else [],
                "modeled_thickness_m": thickness,
                "merge_coplanar_faces": True,
                "wall_plane_pair": planes,
            }
            parts.append(new_part)
            level_rows.append(
                {
                    "name": name,
                    "source_group": f"L{level}_{part['name']}",
                    "source_kind": part["kind"],
                    "axis": "xy"[axis],
                    "thickness_mm": thickness * 1000.0,
                    "thickness_verified": verified,
                    "openings": metadata["openings"],
                    "niches": metadata["niches"],
                    "profile_additions": metadata["profile_additions"],
                    "watertight": True,
                    "volume_m3": abs(float(mesh.volume)),
                    "area_m2": float(mesh.area),
                    "plane_pair": planes,
                }
            )

        floor_region = shapely.set_precision(
            shapely.union_all([top_polygon(part) for part in floor_parts]), 0.0001
        )
        floor_region, removed_holes = remove_tiny_holes(floor_region)
        floor_polys = sorted(
            [poly for poly in polygons(floor_region) if poly.area > 0.0025],
            key=lambda poly: -poly.area,
        )
        floor_rows = []
        for index, poly in enumerate(floor_polys, 1):
            mesh = trimesh.creation.extrude_polygon(poly, height=0.1, engine="earcut")
            mesh.vertices[:, 2] -= 0.1
            mesh.merge_vertices()
            mesh.fix_normals()
            if not mesh.is_watertight:
                raise ValueError(f"L{level} floor plane {index} is not closed")
            common = transform_vertices(mesh.vertices, level, nominal_z, matrix)
            name = f"L{level} floor plane {index:02d} - joined to wall faces"
            parts.append(
                {
                    "name": name,
                    "kind": "floor_wall_joined",
                    "level": level,
                    "colour": [174, 174, 166],
                    "v": np.round(common, 8).tolist(),
                    "f": mesh.faces.tolist(),
                    "merge_coplanar_faces": True,
                    "construction_solid": True,
                }
            )
            floor_rows.append(
                {
                    "name": name,
                    "top_area_m2": float(poly.area),
                    "volume_m3": abs(float(mesh.volume)),
                    "top_local_z_m": 0.0,
                    "top_common_z_range_m": [
                        float(common[:, 2].max()),
                        float(common[:, 2].max()),
                    ],
                    "backing_thickness_m": 0.1,
                    "backing_thickness_verified": False,
                    "holes_over_0_005_m2": len(poly.interiors),
                }
            )

        audit["levels"][str(level)] = {
            "registered_walls_rebuilt": len(wall_parts),
            "measured_thickness_walls": sum(
                part["kind"] == "wall_measured" for part in wall_parts
            ),
            "modeled_thickness_walls": sum(
                part["kind"] == "wall_inferred" for part in wall_parts
            ),
            "new_wall_runs_discovered": 0,
            "wall_parts": level_rows,
            "source_floor_parts_consolidated": len(floor_parts),
            "doorway_thresholds_restored_beyond_clipped_predecessor": restored_thresholds,
            "joined_floor_solids": len(floor_rows),
            "floor_top_area_m2": float(sum(row["top_area_m2"] for row in floor_rows)),
            "pore_holes_removed": removed_holes,
            "floor_parts": floor_rows,
            "source_floor_z_m": manifest["source_floor_z_m"],
            "common_finish_floor_z_m": float(nominal_z + matrix[2, 3]),
        }

    payload = {
        "label": "Soulace | complete-house paired wall planes and joined floor planes",
        "source_native": str(
            (SOURCE / "Soulace_complete_wall_planes_and_floor_junctions.skp").resolve()
        ),
        "expected_base_groups": len(source_audit["groups"]),
        "reference_source_names": sorted(references),
        "parts": parts,
        "render_image_names": [
            "3D",
            "Top",
            "Front",
            "Side",
            "L0 3D",
            "L0 Top",
            "L1 3D",
            "L1 Top",
            "L2 3D",
            "L2 Top",
        ],
        "source_label": (
            "Existing registered wall identities restored as complete planar solids; "
            "measured and modeled thickness confidence retained separately"
        ),
        "note": (
            "L0/L1 wall and floor fragments are hidden references and replaced with continuous planar solids. "
            "No new wall run was discovered. L2 paired walls/floors, stair geometry, beams, and the single cyan "
            "parking/ground plane are preserved. Parking and interior finish floors remain at different elevations. "
            "Modeled wall backs and 100 mm floor backing are explicitly unverified and site accuracy is not certified."
        ),
    }
    additions.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (OUT / "full_house_plane_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "base_groups": payload["expected_base_groups"],
                "references": len(references),
                "new_parts": len(parts),
                "walls": sum(part["kind"] == "wall_paired_planes" for part in parts),
                "floors": sum(part["kind"] == "floor_wall_joined" for part in parts),
                "triangles": sum(len(part["f"]) for part in parts),
                "output": str(additions),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    build()
