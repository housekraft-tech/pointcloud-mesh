"""render_wall_closeup.py
------------------------
Close-up of individual walls cropped straight out of the Poisson mesh, to
confirm the real surface relief (grooves / recesses / skirting / beams)
survives at the per-wall level. Renders each wall three ways:
  <tag>_wallNN_shaded.png  - lit gray surface
  <tag>_wallNN_normal.png  - normal-encoded (colour = surface orientation)
  <tag>_wallNN_depth.png   - depth map (relief pops as gradient/steps)

Uses continuous/measurements.json for each wall's plane (center/dir/normal/
extent) and crops a thin slab of the Poisson mesh around it.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\render_wall_closeup.py \\
      <poisson.obj> <measurements.json> <out_dir> <tag> [n_walls]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

W, H = 1400, 1000


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def render_geom(geom, front, lookat, zoom, light, depth=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=W, height=H)
    vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.06, 0.07, 0.08])
    opt.light_on = light
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_front(front); ctr.set_up([0, 0, 1]); ctr.set_lookat(lookat); ctr.set_zoom(zoom)
    vis.poll_events(); vis.update_renderer()
    if depth:
        d = np.asarray(vis.capture_depth_float_buffer(do_render=True))
        vis.destroy_window()
        return d
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (img * 255).astype(np.uint8)


def main(poisson, mj, out_dir, tag, n_walls=3):
    n_walls = int(n_walls)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    walls = [w for w in json.load(open(mj))["walls"] if w.get("status") == "ok"]
    walls.sort(key=lambda w: -(w["tmax"] - w["tmin"]))          # longest first
    walls = walls[:n_walls]

    log(f"loading {poisson} ...")
    mesh = o3d.io.read_triangle_mesh(str(poisson))
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    if np.argmin(v.max(0) - v.min(0)) == 1:                     # Y-up -> Z-up
        Rz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        mesh.rotate(Rz, center=(0, 0, 0))
        v = np.asarray(mesh.vertices)
    zlo, zhi = v[:, 2].min(), v[:, 2].max()
    log(f"loaded {len(v):,} v; z {zlo:.2f}..{zhi:.2f}")

    # normal-encoded colour copy (colour = n*0.5+0.5)
    n = np.asarray(mesh.vertex_normals)
    ncol = (n * 0.5 + 0.5).clip(0, 1)

    for i, w in enumerate(walls):
        c = np.array(w["center"]); d = np.array(w["dir"]); nm = np.array(w["normal"])
        along = (v[:, 0] - c[0]) * d[0] + (v[:, 1] - c[1]) * d[1]
        perp = (v[:, 0] - c[0]) * nm[0] + (v[:, 1] - c[1]) * nm[1]
        m = (along > w["tmin"] - 0.2) & (along < w["tmax"] + 0.2) & \
            (np.abs(perp) < 0.18) & (v[:, 2] > zlo + 0.02) & (v[:, 2] < zhi - 0.02)
        keep = np.where(m)[0]
        if keep.size < 500:
            log(f"wall {i}: only {keep.size} verts, skip"); continue
        cen = [c[0], c[1], (zlo + zhi) / 2]
        front = [nm[0], nm[1], 0.05]                          # look along wall normal
        zoom = 0.42

        sub = mesh.select_by_index(keep)
        sub.compute_vertex_normals()
        # shaded
        cv2.imwrite(str(out / f"{tag}_wall{i:02d}_shaded.png"),
                    cv2.cvtColor(render_geom(sub, front, cen, zoom, True), cv2.COLOR_RGB2BGR))
        # normal-encoded
        subn = o3d.geometry.TriangleMesh(sub)
        subn.vertex_colors = o3d.utility.Vector3dVector(ncol[keep])
        cv2.imwrite(str(out / f"{tag}_wall{i:02d}_normal.png"),
                    cv2.cvtColor(render_geom(subn, front, cen, zoom, False), cv2.COLOR_RGB2BGR))
        # depth
        dep = render_geom(sub, front, cen, zoom, True, depth=True)
        valid = dep[dep > 0]
        if valid.size:
            lo, hi = np.percentile(valid, [2, 98])
            dn = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
            dn[dep == 0] = 0
            dm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            dm[dep == 0] = (20, 18, 16)
            cv2.imwrite(str(out / f"{tag}_wall{i:02d}_depth.png"), dm)
        log(f"wall {i}: {keep.size:,} verts, len {w['tmax']-w['tmin']:.2f} m -> shaded/normal/depth")
    log("done")


if __name__ == "__main__":
    main(*sys.argv[1:6]) if len(sys.argv) > 5 else main(*sys.argv[1:5])
