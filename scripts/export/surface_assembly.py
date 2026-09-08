"""Assemble the 475 raw patches into surfaces, keeping every piece of evidence.

Implements docs/SURFACE_ASSEMBLY_SPEC.md.

The 253 "surfaces" everything downstream has been built on came from a greedy
merge that never looked at the evidence: 222 reductions, groups of up to 13
patches, and among them groups containing explicit steps and failed fits with
73-85 mm of internal offset. Every model built since inherited that. So this
starts again from the unmerged 475 and refuses to merge anything without cause.

The spatial atom is a SUPPORT COMPONENT -- a connected piece of one patch's
measured 25 mm mask -- not a patch, and never a bounding rectangle. That
distinction matters immediately: what the old pipeline called surface 91 is two
raw patches 8 m apart, and the merge bridged the gap between them.

A pair is judged in this order: whether the sensor saw the two from opposite
sides (then they are two faces of a solid and can never merge, however close);
whether a connected region of their shared support steps; whether one joint
plane fits EVERY member without trimming anyone away; and whether they overlap
or touch enough to be the same surface at all. Any of the first three is a veto,
and a veto is never defeated by a transitive path of positives -- which is why
groups are built by a priority queue with the group-wide tests rerun after each
acceptance, rather than by union-find closure.
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def frame_of(n):
    """A deterministic orthonormal frame on a plane."""
    t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, t)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def sha(path, cap=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(cap))
    return h.hexdigest()[:16]


def components(bins, close=1):
    """8-connected components of a set of occupied bins.

    The closing decides only what counts as ONE piece; it never enters the
    authoritative mask, so no measured area is invented."""
    S = set(map(tuple, bins.tolist()))
    grown = set(S)
    if close:
        for (i, j) in S:
            for di in range(-close, close + 1):
                for dj in range(-close, close + 1):
                    grown.add((i + di, j + dj))
    seen, out = set(), []
    for seed in sorted(grown):
        if seed in seen:
            continue
        stack, comp = [seed], []
        seen.add(seed)
        while stack:
            ci, cj = stack.pop()
            comp.append((ci, cj))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (ci + di, cj + dj)
                    if nb in grown and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        real = sorted(set(comp) & S)
        if real:
            out.append(np.array(real, dtype=np.int64))
    out.sort(key=lambda m: (int(m[:, 0].min()), int(m[:, 1].min()), -len(m)))
    return out


def wfit(P, w):
    """Weighted plane fit in float64, with no trimming of anyone."""
    w = w / w.sum()
    c = (P * w[:, None]).sum(0)
    Q = (P - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    n = vt[-1]
    return n, float(n @ c)


def bin_weights(uv, cell):
    """One vote per occupied bin, shared among the returns inside it."""
    ij = np.floor(uv / cell).astype(np.int64)
    _, inv, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


class Grid:
    """A coarse uniform grid over the LAS, so a component can gather its own
    returns without a global nearest-neighbour search per candidate pair."""

    def __init__(self, P, cell=0.25):
        self.cell = cell
        self.lo = P.min(0)
        key = np.floor((P - self.lo) / cell).astype(np.int64)
        self.dim = key.max(0) + 1
        flat = (key[:, 0] * self.dim[1] + key[:, 1]) * self.dim[2] + key[:, 2]
        order = np.argsort(flat, kind="stable")
        self.order = order
        self.flat = flat[order]

    def box(self, lo, hi):
        a = np.maximum(np.floor((lo - self.lo) / self.cell).astype(np.int64), 0)
        b = np.minimum(np.floor((hi - self.lo) / self.cell).astype(np.int64),
                       self.dim - 1)
        if np.any(a > b):
            return np.zeros(0, np.int64)
        out = []
        for i in range(a[0], b[0] + 1):
            for j in range(a[1], b[1] + 1):
                base = (i * self.dim[1] + j) * self.dim[2]
                s = np.searchsorted(self.flat, base + a[2])
                e = np.searchsorted(self.flat, base + b[2], side="right")
                if e > s:
                    out.append(self.order[s:e])
        return np.concatenate(out) if out else np.zeros(0, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="L0_pointplanes.npz, the 475")
    ap.add_argument("--aligned", required=True, help="npz holding R and origin")
    ap.add_argument("--las", required=True)
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--out", required=True, help="npz; a _manifest.json is written too")
    ap.add_argument("--bin", type=float, default=0.025, help="m: support bin")
    ap.add_argument("--step-bin", type=float, default=0.050, help="m")
    ap.add_argument("--halo", type=float, default=0.030, help="m: lookup halo")
    ap.add_argument("--band", type=float, default=0.030, help="m: association band")
    ap.add_argument("--ang", type=float, default=3.0, help="deg: prefilter only")
    ap.add_argument("--gap", type=float, default=0.050, help="m: prefilter only")
    ap.add_argument("--min-overlap", type=float, default=0.05, help="m2")
    ap.add_argument("--contact", type=float, default=0.30, help="m")
    ap.add_argument("--side", type=float, default=0.90)
    ap.add_argument("--side-band", type=float, default=0.05,
                    help="m: signed distances inside this do not vote")
    ap.add_argument("--max-rms", type=float, default=0.020, help="m")
    ap.add_argument("--fragment", type=float, default=0.01, help="m2")
    ap.add_argument("--step-area", type=float, default=0.10, help="m2")
    a = ap.parse_args()
    cfg = {k: v for k, v in vars(a).items()}

    z = np.load(a.raw)
    N0, D0, RMS, pts0, lab = z["n"], z["d"], z["rms"], z["pts"], z["lab"]
    al = np.load(a.aligned)
    R, origin = np.array(al["R"], float), np.array(al["origin"], float)
    if abs(np.linalg.det(R) - 1) > 1e-9:
        raise SystemExit("R is not a rotation")

    pts = pts0 @ R.T - origin
    N = N0 @ R.T
    D = D0 - N @ origin
    log(f"{len(N0)} raw patches transformed into the aligned frame")

    import laspy
    keep = []
    with laspy.open(a.las) as r:
        for ch in r.chunk_iterator(8_000_000):
            keep.append(np.column_stack([ch.x, ch.y, ch.z]).astype(np.float64))
    P = np.vstack(keep)
    s = np.load(a.sensor)
    traj, fidx = np.array(s["traj"], float), np.array(s["fidx"], float)
    if len(P) != len(fidx):
        raise SystemExit(f"LAS has {len(P):,} points but fidx has {len(fidx):,}")
    if not np.isfinite(P).all():
        raise SystemExit("LAS contains non-finite coordinates")
    log(f"{len(P):,} LAS points, {len(traj):,} trajectory poses")

    # frame validation: transformed SUPPORT against the LAS, not bounding boxes
    tree = cKDTree(P)
    samp = pts[np.linspace(0, len(pts) - 1, 20000).astype(int)]
    dq, _ = tree.query(samp, workers=-1)
    q = np.percentile(dq, [50, 95, 99])
    overlap_axes = bool(np.all(pts.min(0) < P.max(0)) and np.all(pts.max(0) > P.min(0)))
    log(f"frame: support->LAS median {q[0]*1000:.2f} mm, 95th {q[1]*1000:.2f}, "
        f"99th {q[2]*1000:.2f}; bounds overlap every axis: {overlap_axes}")
    if q[0] > a.bin or not overlap_axes:
        raise SystemExit("FRAME MISMATCH: transformed support does not sit on the LAS")

    # ---- 4. support components: the spatial atom ---------------------------
    comps = []
    for p in range(len(N)):
        Pp = pts[lab == p]
        if len(Pp) < 20:
            continue
        n = N[p] / np.linalg.norm(N[p])
        u, v = frame_of(n)
        bins = np.unique(np.floor(np.c_[Pp @ u, Pp @ v] / a.bin).astype(np.int64), axis=0)
        for ci, mask in enumerate(components(bins)):
            area = len(mask) * a.bin * a.bin
            comps.append(dict(raw=int(p), ci=int(ci), n=n, d=float(D[p]), u=u, v=v,
                              mask=mask, area=float(area), rms=float(RMS[p]),
                              fragment=bool(area < a.fragment)))
    log(f"{len(comps)} support components from {len(N)} patches; "
        f"{sum(c['fragment'] for c in comps)} fragments under {a.fragment} m2")

    # ---- 5. full-resolution association, cached once ------------------------
    grid = Grid(P)
    halo_bins = int(np.ceil(a.halo / a.bin))
    for c in comps:
        ctr = (c["mask"] + 0.5) * a.bin
        corners = []
        for uu in (ctr[:, 0].min() - a.halo, ctr[:, 0].max() + a.halo):
            for vv in (ctr[:, 1].min() - a.halo, ctr[:, 1].max() + a.halo):
                corners.append(c["u"] * uu + c["v"] * vv + c["n"] * c["d"])
        corners = np.array(corners)
        cand = grid.box(corners.min(0) - a.band, corners.max(0) + a.band)
        if not len(cand):
            c["idx"] = np.zeros(0, np.int64)
            continue
        Q = P[cand]
        ok = np.abs(Q @ c["n"] - c["d"]) <= a.band
        cand, Q = cand[ok], Q[ok]
        if not len(cand):
            c["idx"] = np.zeros(0, np.int64)
            continue
        ij = np.floor(np.c_[Q @ c["u"], Q @ c["v"]] / a.bin).astype(np.int64)
        want = set(map(tuple, c["mask"].tolist()))
        grown = set()
        for (i, j) in want:
            for di in range(-halo_bins, halo_bins + 1):
                for dj in range(-halo_bins, halo_bins + 1):
                    grown.add((i + di, j + dj))
        sel = np.fromiter((tuple(t) in grown for t in ij.tolist()), bool, len(ij))
        c["idx"] = np.sort(cand[sel])
    log(f"associated {sum(len(c['idx']) for c in comps):,} returns; median "
        f"{np.median([len(c['idx']) for c in comps]):.0f} per component")

    # ---- 6. fractional sensor poses and side evidence -----------------------
    i0 = np.floor(fidx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(traj) - 1)
    alpha = (fidx - i0)[:, None]
    for c in comps:
        c.update(side=0.0, sign=0, reliable=False, nvote=0)
        if len(c["idx"]) < 30:
            continue
        k = c["idx"][np.linspace(0, len(c["idx"]) - 1,
                                 min(4000, len(c["idx"]))).astype(int)]
        sp = (1 - alpha[k]) * traj[i0[k]] + alpha[k] * traj[i1[k]]
        sd = sp @ c["n"] - c["d"]
        vote = np.abs(sd) > a.side_band
        if int(vote.sum()) < 30:
            continue
        pos = float((sd[vote] > 0).mean())
        frac = max(pos, 1 - pos)
        nv = int(vote.sum())
        se = np.sqrt(max(frac * (1 - frac), 1e-6) / nv)
        c.update(side=frac, sign=1 if pos >= 0.5 else -1, nvote=nv,
                 reliable=bool(frac - 1.96 * se > a.side))
    log(f"{sum(c['reliable'] for c in comps)} of {len(comps)} components have a "
        f"reliable sensor side")

    # ---- 7. spatial candidate retrieval ------------------------------------
    NN = np.array([c["n"] for c in comps])
    DD = np.array([c["d"] for c in comps])
    cosang = np.cos(np.radians(a.ang))
    DIL = 2

    def mask_in(ci, cj):
        ctr = (comps[cj]["mask"] + 0.5) * a.bin
        p3 = (comps[cj]["u"] * ctr[:, :1] + comps[cj]["v"] * ctr[:, 1:2]
              + comps[cj]["n"] * comps[cj]["d"])
        return np.floor(np.c_[p3 @ comps[ci]["u"], p3 @ comps[ci]["v"]] / a.bin
                        ).astype(np.int64)

    def spatial(ci, cj):
        A = set(map(tuple, comps[ci]["mask"].tolist()))
        B = set(map(tuple, mask_in(ci, cj).tolist()))
        inter = A & B
        ov = len(inter) * a.bin * a.bin
        if ov >= a.min_overlap:
            return "overlap", ov, 0.0
        gA = set()
        for (i, j) in A:
            for di in range(-DIL, DIL + 1):
                for dj in range(-DIL, DIL + 1):
                    gA.add((i + di, j + dj))
        touch = sorted(gA & B)
        if not touch:
            return None, ov, 0.0
        t = np.array(touch)
        ext = float(max(t[:, 0].max() - t[:, 0].min(),
                        t[:, 1].max() - t[:, 1].min()) * a.bin) if len(t) > 1 else 0.0
        return ("contact" if ext >= a.contact else None), ov, ext

    prefilter = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            dot = float(NN[i] @ NN[j])
            if abs(dot) < cosang:
                continue
            sgn = 1.0 if dot > 0 else -1.0
            if abs(DD[i] - sgn * DD[j]) > a.gap:
                continue
            prefilter.append((i, j, sgn))
    log(f"{len(prefilter)} pairs pass the angle/offset PREFILTER (never evidence)")

    # ---- 8/9/10. joint fit, step test, verdict ------------------------------
    def joint(mem):
        idx = np.unique(np.concatenate([comps[m]["idx"] for m in mem]))
        if len(idx) < 100:
            return None
        Q = P[idx]
        u, v = comps[min(mem)]["u"], comps[min(mem)]["v"]
        n, d = wfit(Q, bin_weights(np.c_[Q @ u, Q @ v], a.bin))
        if n @ comps[min(mem)]["n"] < 0:
            n, d = -n, -d
        per = []
        for m in mem:
            if not len(comps[m]["idx"]):
                per.append(None)
                continue
            r = np.abs(P[comps[m]["idx"]] @ n - d)
            per.append((float(np.median(r)), float(np.percentile(r, 95))))
        return n, d, per, int(len(idx))

    def member_ok(m, st):
        if st is None:
            return False
        noise = max(comps[m]["rms"], 0.001)
        return st[0] <= max(0.005, 1.5 * noise) and st[1] <= max(0.012, 3.0 * noise)

    def medians(m, ci, n, d):
        Q = P[comps[m]["idx"]]
        uv = np.floor(np.c_[Q @ comps[ci]["u"], Q @ comps[ci]["v"]] / a.step_bin
                      ).astype(np.int64)
        dep = Q @ n - d
        key, inv = np.unique(uv, axis=0, return_inverse=True)
        out = {}
        for k in range(len(key)):
            sel = inv == k
            if int(sel.sum()) >= 4:
                out[tuple(key[k].tolist())] = float(np.median(dep[sel]))
        return out

    def step_area(ci, cj, n, d):
        """Compare the two members' MEDIAN depth per shared bin. Never a pooled
        mean -- that cannot tell a step from a thick noise band."""
        if not len(comps[ci]["idx"]) or not len(comps[cj]["idx"]):
            return 0.0, 0.0
        A, B = medians(ci, ci, n, d), medians(cj, ci, n, d)
        shared = sorted(set(A) & set(B))
        if not shared:
            return 0.0, 0.0
        tol = max(0.010, 3 * np.sqrt(comps[ci]["rms"] ** 2 + comps[cj]["rms"] ** 2))
        bad = {k for k in shared if abs(A[k] - B[k]) > tol}
        if not bad:
            return 0.0, 0.0
        seen, biggest = set(), 0
        for seed in sorted(bad):
            if seed in seen:
                continue
            stack, size = [seed], 0
            seen.add(seed)
            while stack:
                x, y = stack.pop()
                size += 1
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        nb = (x + di, y + dj)
                        if nb in bad and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            biggest = max(biggest, size)
        cell = a.step_bin * a.step_bin
        return biggest * cell, len(bad) * cell

    REL = {}
    for (i, j, sgn) in prefilter:
        kind, ov, ext = spatial(i, j)
        rec = dict(a=int(i), b=int(j),
                   angle=float(np.degrees(np.arccos(min(1.0, abs(NN[i] @ NN[j]))))),
                   offset=float(abs(DD[i] - sgn * DD[j])),
                   spatial=kind, overlap_m2=float(ov), contact_m=float(ext))
        if not comps[i]["reliable"] or not comps[j]["reliable"]:
            rec["rel"] = "side_ambiguous"
        elif comps[i]["sign"] * comps[j]["sign"] * (1 if sgn > 0 else -1) < 0:
            rec["rel"] = "opposite_faces"
        elif kind is None:
            rec["rel"] = "spatially_unrelated"
        else:
            J = joint([i, j])
            if J is None:
                rec["rel"] = "joint_fit_failed"
            else:
                n, d, per, nidx = J
                big, tot = step_area(i, j, n, d)
                rec.update(step_largest_m2=float(big), step_total_m2=float(tot),
                           joint=[list(p) if p else None for p in per], n_idx=nidx)
                if big > a.step_area:
                    rec["rel"] = "step"
                elif not (member_ok(i, per[0]) and member_ok(j, per[1])):
                    rec["rel"] = "joint_fit_failed"
                elif max(comps[i]["rms"], comps[j]["rms"]) > a.max_rms:
                    rec["rel"] = "side_ambiguous"
                else:
                    rec["rel"] = "merge_candidate"
        REL[(i, j)] = rec
    tally = defaultdict(int)
    for r in REL.values():
        tally[r["rel"]] += 1
    log("relations: " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))

    VETO = ("opposite_faces", "step", "joint_fit_failed", "side_ambiguous")

    # ---- 11. grouping: priority queue, no transitive closure ---------------
    group = {i: i for i in range(len(comps))}
    members = {i: [i] for i in range(len(comps))}

    def rel(x, y):
        return REL.get((min(x, y), max(x, y)))

    def has_veto(ga, gb):
        return any((rel(x, y) or {}).get("rel") in VETO
                   for x in members[ga] for y in members[gb])

    def has_link(ga, gb):
        return any((rel(x, y) or {}).get("rel") == "merge_candidate"
                   for x in members[ga] for y in members[gb])

    accepted = 0
    while True:
        best = None
        roots = sorted(members)
        for ai in range(len(roots)):
            for bi in range(ai + 1, len(roots)):
                ga, gb = roots[ai], roots[bi]
                if not has_link(ga, gb) or has_veto(ga, gb):
                    continue
                mem = sorted(members[ga] + members[gb])
                J = joint(mem)
                if J is None:
                    continue
                _, _, per, _ = J
                if not all(member_ok(m, p) for m, p in zip(mem, per)):
                    continue
                sg = {comps[m]["sign"] for m in mem if comps[m]["reliable"]}
                if len(sg) > 1:
                    continue
                key = (-max(p[1] for p in per if p), tuple(mem))
                if best is None or key > best[0]:
                    best = (key, ga, gb, mem)
        if best is None:
            break
        _, ga, gb, mem = best
        root = min(ga, gb)
        for m in mem:
            group[m] = root
        members[root] = mem
        if root != ga:
            members.pop(ga, None)
        if root != gb:
            members.pop(gb, None)
        accepted += 1
    log(f"{accepted} unions accepted; {len(members)} surfaces from {len(comps)} components")

    # ---- 11b. one final fit per group, from sorted inputs -------------------
    rows = []
    for root in sorted(members):
        mem = sorted(members[root])
        J = joint(mem)
        if J is None:
            n, d, per = comps[mem[0]]["n"], comps[mem[0]]["d"], [None] * len(mem)
        else:
            n, d, per, _ = J
        rows.append(dict(group=int(root), members=[int(m) for m in mem],
                         raws=sorted({comps[m]["raw"] for m in mem}),
                         n=[float(x) for x in n], d=float(d),
                         area=float(sum(comps[m]["area"] for m in mem)),
                         per_member=[list(p) if p else None for p in per]))

    off = np.cumsum([0] + [len(c["mask"]) for c in comps])
    np.savez_compressed(
        a.out,
        raw=np.array([c["raw"] for c in comps]),
        ci=np.array([c["ci"] for c in comps]),
        n=np.array([c["n"] for c in comps]), d=np.array([c["d"] for c in comps]),
        area=np.array([c["area"] for c in comps]),
        side=np.array([c["side"] for c in comps]),
        sign=np.array([c["sign"] for c in comps]),
        reliable=np.array([c["reliable"] for c in comps]),
        group=np.array([group[i] for i in range(len(comps))]),
        mask_off=off, mask=np.vstack([c["mask"] for c in comps]),
        gn=np.array([r["n"] for r in rows]), gd=np.array([r["d"] for r in rows]),
        groups=np.array([r["group"] for r in rows]))
    json.dump(dict(schema="surface_assembly/1", config=cfg,
                   inputs={k: sha(v) for k, v in dict(raw=a.raw, las=a.las,
                                                      sensor=a.sensor).items()},
                   n_raw=int(len(N)), n_components=len(comps),
                   frame_check_mm=[float(x * 1000) for x in q],
                   relations=list(REL.values()), groups=rows),
              open(a.out.replace(".npz", "_manifest.json"), "w"), indent=1)
    log(f"-> {a.out} and its manifest")

    # ---- 13. acceptance ----------------------------------------------------
    fails = []
    total_bins = sum(len(c["mask"]) for c in comps)
    claimed = sum(len(comps[m]["mask"]) for r in rows for m in r["members"])
    if claimed != total_bins:
        fails.append(f"conservation: {claimed} bins claimed of {total_bins}")
    for r in rows:
        mem = r["members"]
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                rr = rel(mem[x], mem[y])
                if rr and rr["rel"] in VETO:
                    fails.append(f"group {r['group']} contains a {rr['rel']} pair "
                                 f"({mem[x]},{mem[y]})")
        for m, p in zip(mem, r["per_member"]):
            if len(mem) > 1 and p and not member_ok(m, tuple(p)):
                fails.append(f"group {r['group']} member {m} fails the final fit")
    print()
    if fails:
        print(f"ACCEPTANCE: FAIL ({len(fails)} problems)")
        for f in fails[:12]:
            print("  " + f)
        raise SystemExit(1)
    print("ACCEPTANCE: PASS — conservation holds, no group contains a veto pair, "
          "and every member passes its group's final fit")


if __name__ == "__main__":
    main()
