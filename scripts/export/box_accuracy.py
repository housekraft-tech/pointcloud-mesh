"""How far is the box model from what was scanned -- both ways.

One direction is not enough. Distance from the scan to the boxes says whether
anything measured was left out; distance from the boxes to the scan says
whether anything was invented. A model of paper-thin sheets scores perfectly on
the first and terribly on the second, and a model that fills the building with
solid mass does the opposite.

Reported per kind, and per part so the worst offenders can be named.  The
optional JSON and Markdown reports turn those distances into an explicit
tolerance gate for the SketchUp handover.  Passing this gate proves internal
scan-to-model agreement; it does NOT prove absolute accuracy until independent
site controls (tape/total-station/check dimensions) pass the same tolerance.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, trimesh, open3d as o3d

KINDS = {"wall": ("wall", "column", "parapet"),
         "slab": ("floor", "ceiling", "dropped_ceiling", "beam")}


def distance_stats(values_mm, tolerance_mm):
    """Summarise distances and grade them against ``tolerance_mm``.

    p95 is the pass criterion: a handful of inevitable edge/outlier samples
    cannot veto a part, but a broad misfit cannot hide behind a good median.
    ``review`` means the typical surface is inside tolerance while too much of
    the surface is not.  ``fail`` means even the median misses the target.
    """
    values = np.asarray(values_mm, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0, "median_mm": None, "p90_mm": None, "p95_mm": None,
            "max_mm": None, "within_tolerance_pct": 0.0, "status": "fail",
        }
    median = float(np.median(values))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))
    if p95 <= tolerance_mm:
        status = "pass"
    elif median <= tolerance_mm:
        status = "review"
    else:
        status = "fail"
    return {
        "count": int(values.size),
        "median_mm": median,
        "p90_mm": p90,
        "p95_mm": p95,
        "max_mm": float(values.max()),
        "within_tolerance_pct": float(100.0 * np.mean(values <= tolerance_mm)),
        "status": status,
    }


def overall_surface_status(kind_stats):
    """Combine wall/slab grades without letting a missing kind pass."""
    statuses = [row.get("status", "fail") for row in kind_stats.values()]
    if statuses and all(s == "pass" for s in statuses):
        return "pass"
    if statuses and all(s in ("pass", "review") for s in statuses):
        return "review"
    return "fail"


def thickness_evidence(manifest):
    """Count measured and inferred wall thicknesses in a modular manifest."""
    walls = [p for p in manifest.get("parts", []) if p.get("kind") in ("wall", "parapet")]
    measured = [p for p in walls if p.get("solid_thickness_measured") is True]
    inferred = [p for p in walls if p not in measured]
    return {
        "total_walls": len(walls),
        "measured_walls": len(measured),
        "inferred_or_unmeasured_walls": len(inferred),
        "measured_pct": float(100.0 * len(measured) / max(len(walls), 1)),
        "inferred_or_unmeasured_names": [p.get("name", "?") for p in inferred],
    }


def _fmt(value):
    return "-" if value is None else f"{value:.1f}"


def markdown_report(report):
    """Render the JSON report as a compact designer-facing audit."""
    tol = report["tolerance_mm"]
    absolute_label = (
        "NOT YET VERIFIED"
        if report["absolute_accuracy_status"].startswith("not_verified")
        else report["absolute_accuracy_status"].replace("_", " ").upper()
    )
    lines = [
        f"# {tol:g} mm accuracy gate",
        "",
        f"- Release outcome: **{report['release_status'].upper()}**",
        f"- Surface-fit outcome: **{report['surface_fit_status'].upper()}**",
        f"- Every-room outcome: **{report['room_accuracy']['status'].upper()}**",
        f"- Absolute accuracy: **{absolute_label}**",
        "- Scan-to-model agreement is an internal consistency check. Absolute "
        "accuracy requires independent site control dimensions within the same tolerance.",
        "",
        "## Surface fit",
        "",
        "| direction | kind | status | median mm | p90 mm | p95 mm | within tolerance |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for kind, row in report["scan_to_box"].items():
        lines.append(
            f"| scan-derived surface to model | {kind} | {row['status']} | "
            f"{_fmt(row['median_mm'])} | {_fmt(row['p90_mm'])} | {_fmt(row['p95_mm'])} | "
            f"{row['within_tolerance_pct']:.1f}% |"
        )
    row = report["box_to_scan_support"]
    lines.append(
        f"| model to scan support* | all | support only | {_fmt(row['median_mm'])} | "
        f"{_fmt(row['p90_mm'])} | {_fmt(row['p95_mm'])} | "
        f"{row['within_tolerance_pct']:.1f}% |"
    )
    ev = report["wall_thickness_evidence"]
    lines += [
        "",
        "*Hidden end, top and junction faces are physically unscannable, so model-to-scan "
        "support is reported but does not decide the tolerance gate.",
        "",
        "## Every-room gate",
        "",
        f"- Rule: {report['room_accuracy']['criterion']}",
        f"- Status: {report['room_accuracy']['status']}",
        f"- Detail: {report['room_accuracy'].get('reason', 'See room report.')}",
        "",
        "## Wall-thickness evidence",
        "",
        f"- Measured from both faces: {ev['measured_walls']}/{ev['total_walls']} "
        f"({ev['measured_pct']:.1f}%)",
        f"- Inferred or unmeasured: {ev['inferred_or_unmeasured_walls']}",
    ]
    if ev["inferred_or_unmeasured_names"]:
        lines.append("- Review before fabrication: " + ", ".join(ev["inferred_or_unmeasured_names"]))
    lines += [
        "",
        "## Per-part scan-derived surface to model",
        "",
        "| part | kind | thickness evidence | status | median mm | p95 mm | within tolerance |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for part in sorted(report["parts"], key=lambda p: (-p["p95_mm"], p["name"])):
        lines.append(
            f"| {part['name']} | {part['kind']} | {part['thickness_evidence']} | "
            f"{part['status']} | {part['median_mm']:.1f} | {part['p95_mm']:.1f} | "
            f"{part['within_tolerance_pct']:.1f}% |"
        )
    lines += [
        "",
        "## Release rule",
        "",
        f"A part may be labelled scan-consistent at +/-{tol:g} mm only when its status is "
        "`pass`. A wall thickness may be labelled measured only when both faces were observed. "
        "All other parts stay visible but must be marked for review; they must not silently enter "
        "fabrication or quantity decisions as exact geometry.",
        "",
    ]
    return "\n".join(lines)


def scene(m):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(m.vertices), o3d.core.float32),
        o3d.core.Tensor(np.asarray(m.faces), o3d.core.uint32)))
    return s


def dist(sc, P):
    return sc.compute_distance(o3d.core.Tensor(np.asarray(P, np.float32))).numpy()*1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="the modular directory")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--boxes", required=True, help="boxes_union.glb")
    ap.add_argument("--parts", type=int, default=10, help="how many worst parts to name")
    ap.add_argument("--tolerance-mm", type=float, default=10.0,
                    help="p95 scan-to-model tolerance used by the pass gate")
    ap.add_argument("--report-json", default=None,
                    help="optional machine-readable accuracy report")
    ap.add_argument("--report-md", default=None,
                    help="optional designer-readable accuracy report")
    ap.add_argument("--room-report", default=None,
                    help="optional JSON from room_accuracy.py; without it the "
                         "release gate cannot pass")
    ap.add_argument("--require-pass", action="store_true",
                    help="exit non-zero unless surfaces and every controlled "
                         "room dimension pass")
    a = ap.parse_args()

    if a.tolerance_mm <= 0:
        ap.error("--tolerance-mm must be positive")

    man = json.load(open(f"{a.dir}/manifest.json"))
    zf = man["floor_z"]; zc = zf + man["modal_ceiling_height_mm"]/1000.0
    V = np.load(f"{a.dir}/verts.npy"); T = np.load(a.cache)["T"]
    lab = np.load(f"{a.dir}/labels.npy"); names = json.load(open(f"{a.dir}/names.json"))
    box = trimesh.load(a.boxes); box = box.to_mesh() if hasattr(box, "to_mesh") else box
    scan = trimesh.Trimesh(V, T[lab >= 0], process=False)
    sb, ss = scene(box), scene(scan)

    band = lambda P: P[(P[:, 2] > zf-0.7) & (P[:, 2] < zc+0.25)]
    kind_stats = {}
    for ki, (k, pref) in enumerate(KINDS.items()):
        keep = np.isin(lab, [i for i, n in enumerate(names) if n.startswith(pref)])
        if keep.sum() < 500:
            continue
        P = band(trimesh.sample.sample_surface(
            trimesh.Trimesh(V, T[keep], process=False), 150000, seed=100 + ki)[0])
        d = dist(sb, P)
        stats = distance_stats(d, a.tolerance_mm)
        kind_stats[k] = stats
        print(f"  scan->box {k:5s}: {stats['status']:6s}  median {stats['median_mm']:5.1f}  "
              f"p95 {stats['p95_mm']:6.1f} | <={a.tolerance_mm:g}mm "
              f"{stats['within_tolerance_pct']:4.1f}%")
    Q = band(trimesh.sample.sample_surface(box, 150000, seed=200)[0])
    d = dist(ss, Q)
    support_stats = distance_stats(d, a.tolerance_mm)
    print(f"  box->scan      : support  median {support_stats['median_mm']:5.1f}  "
          f"p95 {support_stats['p95_mm']:6.1f} | <={a.tolerance_mm:g}mm "
          f"{support_stats['within_tolerance_pct']:4.1f}%   "
          f"{len(box.faces)} tris, {box.volume:.1f} m3, "
          f"watertight {box.is_watertight}")

    parts_by_name = {p["name"]: p for p in man["parts"]}
    part_rows = []
    for i, n in enumerate(names):
        if (lab == i).sum() < 400:
            continue
        m = trimesh.Trimesh(V, T[lab == i], process=False)
        d = dist(sb, trimesh.sample.sample_surface(m, 3000, seed=1000 + i)[0])
        stats = distance_stats(d, a.tolerance_mm)
        meta = parts_by_name.get(n, {})
        if meta.get("kind") in ("wall", "parapet"):
            evidence = "measured" if meta.get("solid_thickness_measured") is True else "inferred/unmeasured"
        else:
            evidence = "n/a"
        part_rows.append({
            "name": n, "kind": meta.get("kind", "?"),
            "area_m2": float(m.area), "thickness_evidence": evidence, **stats,
        })

    if a.parts:
        rows = sorted(part_rows, key=lambda r: -r["median_mm"] * r["area_m2"])
        print(f"  worst parts (median error x area):")
        for row in rows[:a.parts]:
            print(f"    {row['median_mm']:7.1f} mm over {row['area_m2']:5.1f} m2   "
                  f"{row['name']} ({row['kind']}, {row['status']})")

    room_accuracy = {
        "status": "not_evaluated",
        "criterion": (
            f"every named room width, length and clear height must be within "
            f"+/-{a.tolerance_mm:g} mm of independent site controls"
        ),
        "reason": "no --room-report supplied",
    }
    if a.room_report:
        room_accuracy = json.loads(Path(a.room_report).read_text(encoding="utf-8"))
        room_accuracy.setdefault(
            "criterion",
            f"every room dimension must be within +/-{a.tolerance_mm:g} mm",
        )

    surface_status = overall_surface_status(kind_stats)
    release_status = (
        "pass" if surface_status == "pass" and room_accuracy.get("status") == "pass"
        else "fail"
    )
    report = {
        "schema_version": 1,
        "tolerance_mm": float(a.tolerance_mm),
        "release_status": release_status,
        "surface_fit_status": surface_status,
        "absolute_accuracy_status": (
            "verified_for_controlled_room_dimensions"
            if room_accuracy.get("status") == "pass"
            else "not_verified_without_passing_independent_room_controls"
        ),
        "room_accuracy": room_accuracy,
        "scan_to_box": kind_stats,
        "box_to_scan_support": support_stats,
        "wall_thickness_evidence": thickness_evidence(man),
        "model": {
            "triangles": int(len(box.faces)), "volume_m3": float(box.volume),
            "watertight": bool(box.is_watertight),
        },
        "parts": part_rows,
    }
    if a.report_json:
        path = Path(a.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report JSON     : {path}")
    if a.report_md:
        path = Path(a.report_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(report), encoding="utf-8")
        print(f"  report Markdown : {path}")
    if a.require_pass and report["release_status"] != "pass":
        raise SystemExit(
            f"accuracy gate failed: wall/slab p95 must be <= {a.tolerance_mm:g} mm "
            "and every controlled room dimension must pass"
        )


if __name__ == "__main__":
    main()
