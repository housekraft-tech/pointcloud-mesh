# Raw-Patch Surface Assembly Specification

Status: implementation contract for the L0 prototype.

This stage replaces both the old greedy `merge_planes.py` result and the
pairwise-union behavior currently used by `kinetic_cells.py`. It starts from the
475 raw patches in `L0_pointplanes.npz`. The 253-surface `L0_merged.npz` and
`L0_aligned.npz` geometry must not be used as assembly input.

The output of this stage is evidence-preserving planar surface data. It is not a
wall model, solid model, room model, or permission to extend a surface beyond
measured support.

## 1. Required inputs

- Raw plane file: arrays `n`, `d`, `rms`, `count`, `pts`, and `lab` from the
  unmerged 475-patch extraction.
- Aligned LAS in the same point order used to create `L0_sensor.npz`.
- Sensor file containing `traj`, fractional `fidx`, and `ts`.
- The exact `R` and `origin` stored in the existing `L0_aligned.npz`, used only
  as a coordinate-frame transform.
- Declared configuration values and input hashes in the output manifest.

The implementation must fail closed if the LAS point count differs from the
length of `fidx`, an input contains non-finite values, plane normals are not
unit length within tolerance, or the frame validation below fails.

## 2. Coordinate frame

Reuse the existing `R` and `origin`. Do not re-estimate or silently update the
building frame in this stage.

For row-vector points and normals:

```text
p_aligned = p_raw @ R.T - origin
n_aligned = n_raw @ R.T
d_aligned = d_raw - dot(n_aligned, origin)
```

The stored transform is rigid (`det(R) = 1`; measured orthogonality error is
approximately `1.1e-16`), so it cannot change angles, inter-plane distances,
joint-fit residuals, support relationships, or sensor-side relationships.

For audit only, report the grid yaw estimated from the raw patches. On the
current files it differs from the stored yaw by about `0.0616 deg`. This report
must not change the output frame.

Frame validation must compare transformed raw support points with the aligned
LAS, not merely compare sensor and LAS bounding boxes. Report nearest-LAS
distance quantiles for a deterministic support sample and fail if the result is
incompatible with the support voxel size. Also verify that transformed support
and LAS bounds overlap on every axis.

## 3. Terms and identities

### Raw patch

One original label from `L0_pointplanes.npz`. Its ID is permanent provenance.

### Support component (the spatial evidence atom)

A connected component of a raw patch's measured 2-D support. All spatial
retrieval, overlap, adjacency, side evidence, and downstream provenance operate
on support components, not bounding rectangles and not an undifferentiated
plane family.

Every support component has a stable ID `(raw_patch_id, component_index)` and
retains its parent raw patch ID. Component indices are assigned by a stable
sort of their minimum support-bin coordinate, then area.

### Surface group

A set of support components allowed to share one canonical plane equation after
all pair and group tests pass. Grouping changes the common plane parameters; it
does not union, convex-hull, fill, or bridge their support masks.

### Relation

An immutable, symmetric evidence record between two support components. A
relation is one of:

- `merge_candidate`
- `opposite_faces`
- `step`
- `joint_fit_failed`
- `side_ambiguous`
- `spatially_unrelated`

All metrics and thresholds that produced the relation are stored.

## 4. Measured support and component splitting

Project each raw patch to a deterministic orthonormal `(u, v)` frame and mark
occupied 25 mm support bins. Store the occupied mask sparsely (sorted integer
bin coordinates or row-run encoding). A polygon representation may also be
stored, but the sparse mask is authoritative.

Use 8-connectivity after a one-bin morphological closing for connectivity only.
The closing must not add measured area to the authoritative mask. Preserve
holes in the mask.

If the connectivity operation yields multiple material components, create
multiple support-component records. They may retain the same parent plane fit,
but they remain separate spatial atoms throughout the pipeline. Never replace
them with one bounding rectangle or convex hull.

Small fragments are retained and flagged rather than silently discarded.
Default `fragment_area` is `0.01 m2`. A fragment cannot initiate a merge and is
never absorbed merely because another component has the same raw parent; it may
join a group only through independently valid spatial and fit evidence.

The old merged surface 91 is not a single raw patch with two clusters. It maps
to raw patches 268 and 269, with 6,008 and 4,067 support points respectively.
They must begin as separate atoms. Old surface 87 likewise maps to raw patches
452 and 473 and must begin as separate atoms.

## 5. Full-resolution point association

Build the LAS spatial index once. Cache the LAS indices associated with each
support component once; do not repeat a global nearest-neighbor search for every
candidate pair.

Association uses the component's actual support mask, expanded only by a
declared lookup halo, plus a fixed plane-distance association band. These gates
define membership and must be reported. They are not adaptive outlier trimming.

The same LAS index may appear in the membership cache of multiple raw
components. A group fit must use the sorted unique union of LAS indices, so no
return receives duplicate weight. Per-member residual tests use that member's
own cached index set.

If full-resolution association cannot be made reliable, fail or explicitly
fall back to raw support points with `point_source = support_sample`; never call
nearest-one-point-per-seed data "full resolution."

## 6. Fractional sensor poses and side evidence

For every LAS return index, interpolate the sensor position from fractional
`fidx`:

```text
i0 = floor(fidx)
i1 = min(i0 + 1, len(traj) - 1)
alpha = fidx - i0
sensor = (1 - alpha) * traj[i0] + alpha * traj[i1]
```

Do not cast `fidx` directly to integer. Verify its range and the one-to-one
point-order contract with the LAS. Store the input `ts` range and trajectory
metadata even though interpolation is indexed by `fidx`.

For each component, compute signed sensor distances using its plane normal.
Exclude samples whose signed distance is inside the declared pose/plane
uncertainty band; they do not vote. Report positive, negative, excluded, and
zero counts and a confidence interval for the majority fraction.

A component is side-reliable only when its lower confidence bound exceeds the
configured side threshold. A pair can be `opposite_faces` only when both
components are individually side-reliable and their reliable signs oppose.
Ambiguity is evaluated before the opposite-face verdict.

Orient a canonical surface normal so its reliable sensor/free side is positive.
If no reliable side exists, use a deterministic lexicographic normal sign and
mark orientation as ambiguous.

## 7. Spatial candidate retrieval

Candidate retrieval applies to all orientations, not only vertical planes.
Angle and offset gates alone are insufficient.

Project both component masks into a common candidate frame. A pair is spatially
eligible only if at least one of the following holds:

1. their measured support masks overlap by a configured minimum area; or
2. masks dilated by at most two support bins touch and the resulting contact
   extends for at least `0.30 m`.

Record measured overlap area, minimum support distance, and contact extent.
Bounding-box overlap is only an index prefilter and is never merge evidence.

Components that are neither overlapping nor locally adjacent are
`spatially_unrelated`. They are not merged merely because their equations are
similar. Later modeling may infer that two such components belong to one wall;
surface assembly may not.

## 8. Joint plane fit

Use the sorted unique union of the components' associated full-resolution LAS
indices. Fit in float64.

Area weighting is computed once over the union: every occupied 25 mm support
bin has total weight one, regardless of return count and regardless of how many
members claim a return. Do not compute a separate unit vote for each member in
the same bin.

Do not iteratively trim points after membership has been established. The
output records the fitted normal, offset, all membership gates, effective
support-bin count, and per-member residuals.

Every member must pass both:

```text
median <= max(5 mm, 1.5 * member_noise)
p95    <= max(12 mm, 3.0 * member_noise)
```

`member_noise` is the declared measurement/noise estimate, not an unlimited
license derived from a visibly mixed patch. Components above a configured
maximum admissible RMS are ambiguous and cannot initiate merging.

## 9. Step test

The step test compares members; it never measures the pooled mean.

For each shared 50 mm bin having the minimum required observations from each
member:

1. compute each member's median signed depth in the common candidate frame;
2. compute the absolute difference between those medians;
3. mark the bin stepped when the difference exceeds
   `max(10 mm, 3 * sqrt(noise_i^2 + noise_j^2))`;
4. form 8-connected components of stepped bins; and
5. veto merging when any connected stepped component has area greater than
   `0.10 m2`.

For merely adjacent masks, apply the same median-depth comparison to matched
boundary strips inside the declared contact halo. If neither shared bins nor a
valid boundary-strip comparison exists, there is no step evidence—but the pair
also lacks sufficient spatial evidence to merge.

Store per-bin/member counts, the largest connected step area, and the total
stepped area for audit.

## 10. Pair verdict order

Evaluate pair verdicts in this order:

1. invalid or insufficient evidence -> `side_ambiguous` or an explicit failure;
2. two reliable opposed side signs -> `opposite_faces`;
3. connected step above the gate -> `step`;
4. failed per-member joint residual -> `joint_fit_failed`;
5. valid spatial relation plus all tests passed -> `merge_candidate`;
6. otherwise -> `spatially_unrelated`.

`opposite_faces`, `step`, `joint_fit_failed`, and `side_ambiguous` are merge
vetoes. A veto is never weakened by a transitive positive path.

## 11. Group assembly

Do not take the connected components or unconstrained union-find closure of
pairwise merge edges.

Start with singleton groups. A proposed group union is admissible only if:

- no cross-group pair has a veto relation;
- the union is connected through valid spatial merge-candidate relations;
- a fresh group-wide fit passes every original member;
- every reliable member has the same sensor/free-side orientation; and
- all stored pairwise step vetoes remain absent.

Use a deterministic global priority queue of admissible unions. Rank by a
documented evidence margin; break ties lexicographically by sorted component
IDs. After every accepted union, recompute group-wide evidence for affected
proposals. This makes results independent of input/candidate enumeration order.

After grouping is final, recompute every canonical plane once from the sorted
original component IDs and sorted unique LAS indices. Do not retain an
incrementally accumulated plane fit as the final result.

## 12. Required outputs

Write a numeric artifact plus a human-readable JSON manifest. Together they
must contain:

- input paths, hashes, point counts, transform, and all configuration values;
- every raw patch and support-component ID;
- authoritative sparse support masks and component areas;
- cached point-membership offsets/indices or a separately hashed cache;
- every evaluated pair relation and its complete metrics;
- final group membership;
- canonical plane parameters and per-member fit metrics;
- reliable side sign/confidence or an explicit ambiguity flag;
- all merge vetoes affecting each group;
- deterministic schema/version identifiers.

No output support may be represented solely by `umin`, `umax`, `vmin`, and
`vmax`.

## 13. Acceptance test

One audit command must exit nonzero unless all of these hold:

1. **Conservation:** every raw patch and every authoritative measured support bin
   appears exactly once in the component lineage. No measured area is invented.
2. **No forbidden closure:** no final group contains an opposite, step,
   failed-fit, or side-ambiguous pair.
3. **Group validity:** a fresh final fit passes every original member of every
   group.
4. **Spatial validity:** every non-singleton group is connected through actual
   overlap/contact relations; no bounding-box-only edge is used.
5. **Disconnected-support safety:** the raw descendants of old surfaces 91 and
   87 remain distinct support components and no support mask fills their gaps.
6. **No double counting:** each group fit uses unique LAS indices and unit total
   weight per occupied support bin.
7. **Frame validity:** transformed raw support is consistent with the aligned
   LAS; the old unaligned-plane/aligned-LAS combination fails this test.
8. **Shuffle invariance:** at least ten deterministic shuffles of patch and
   candidate enumeration produce identical sets of sorted group memberships,
   identical component lineage and support masks, normal differences below
   `1e-8 rad`, and offset/RMS differences below `1e-6 m`.

Bitwise-identical floating-point plane parameters are not required. Bitwise
identity is brittle across BLAS implementations and thread counts. Exact group
membership and lineage are required; canonical geometry must agree well below
measurement resolution.

There is no required final surface count, and the seven legacy 253-surface pair
verdicts are diagnostics rather than ground truth. In particular, a pair with
no valid overlap or contact evidence must not be forced to merge to preserve an
old expected result.

## 14. Optional preflight sensitivity dump

No additional experiment is required before implementation. One inexpensive
diagnostic is recommended once component extraction exists:

- repeat support connectivity at 20, 25, 40, and 50 mm bins;
- repeat with zero-, one-, and two-bin connectivity closing;
- report components whose count or largest-area fraction changes;
- report pair candidate counts by overlap, contact, and bbox-only categories.

This dump should classify unstable support topology as ambiguous; it should not
be used to select whichever setting produces the most merges.

## 15. Reuse policy for existing scripts

Write `surface_assembly.py` with fresh orchestration and explicit stages.

The following ideas may be retained as small, tested pure functions:

- deterministic plane-frame construction;
- weighted PCA/SVD plane fitting;
- angle and normalized-offset calculations;
- one-vote-per-support-bin weighting, corrected to operate on the unique group
  union;
- JSON evidence formatting.

Do not reuse the existing candidate loop, frame check, integer sensor lookup,
pooled step test, verdict order, repeated nearest-neighbor lookup, or union-find
grouping. Nothing from `kinetic_cells.py` is needed by this assembly stage.
