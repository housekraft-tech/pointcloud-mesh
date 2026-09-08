import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/export'))
from run_architectural_flow import rigid, transform_part, validate_parts, validate_bounded_walls, read_manifest, read_registered_points


def cube(name='wall'):
    mesh = trimesh.creation.box(extents=(.2, 2, 3))
    return {'name': name, 'kind': 'wall', 'v': mesh.vertices.tolist(), 'f': mesh.faces.tolist()}


@pytest.mark.parametrize('matrix', [np.diag([1000,1000,1000,1]), np.diag([-1,1,1,1]), np.full((4,4), np.nan)])
def test_registration_rejects_scale_reflection_and_nan(matrix):
    with pytest.raises(ValueError):
        rigid(matrix)


def test_plane_and_vertices_remain_consistent_after_level_registration():
    part = cube()
    part['wall_plane_pair'] = [{'normal':[1,0,0], 'offset_m':.1, 'side':'A'}]
    matrix = np.array([[0,-1,0,4],[1,0,0,-3],[0,0,1,6.5],[0,0,0,1]])
    result = transform_part(part, matrix, name='upper wall', level=2)
    plane = result['wall_plane_pair'][0]
    source = np.asarray(part['v'])
    side = np.isclose(source[:,0], .1)
    actual = np.asarray(result['v'])[side] @ plane['normal']
    assert np.allclose(actual, plane['offset_m'])
    assert result['name'] == 'upper wall' and result['level'] == 2
    assert np.isclose(np.ptp(np.asarray(result['v'])[:,2]), 3)


def test_duplicate_parts_cannot_silently_overwrite_native_groups():
    with pytest.raises(ValueError, match='duplicate'):
        validate_parts([cube(), cube()])


def test_face_indices_must_be_integer_and_in_range():
    part = cube(); part['f'][0][0] = -1
    with pytest.raises(ValueError, match='index'):
        validate_parts([part])


def test_unknown_units_fail_before_geometry_output(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'units':'millimetres'}))
    with pytest.raises(ValueError, match='units'):
        read_manifest(path)


def test_streamed_las_retains_original_samples_and_applies_declared_registration(tmp_path):
    import laspy
    las = laspy.LasData(laspy.LasHeader(point_format=3, version='1.2'))
    las.header.scales = [.001,.001,.001]
    las.x = [.001,.002,1.]; las.y = [0,0,0]; las.z = [0,0,0]
    path = tmp_path / 'scan.las'; las.write(path)
    matrix = np.eye(4); matrix[:3,3] = [2,3,4]
    points, records = read_registered_points(
        [{'id':'test','path':str(path),'scan_to_model':matrix.tolist()}],
        np.array([[1,2,3],[4,4,5]]), .02)
    assert len(points) == 2
    assert any(np.allclose(p, [2.001,3,4]) for p in points)
    assert records[0]['input_returns'] == 3
    assert records[0]['roi_returns'] == 3
    assert records[0]['evidence_points'] == 2


def test_empty_registered_reference_is_a_frame_error(tmp_path):
    import laspy
    las = laspy.LasData(laspy.LasHeader(point_format=3, version='1.2'))
    las.x=[0];las.y=[0];las.z=[0]
    path=tmp_path/'scan.las';las.write(path)
    with pytest.raises(ValueError, match='check registration'):
        read_registered_points([{'id':'test','path':str(path),'scan_to_model':np.eye(4).tolist()}],
                               np.array([[10,10,10],[11,11,11]]))


def test_unverified_wall_solid_cannot_bypass_raw_support_clipping():
    with pytest.raises(ValueError, match='omit current_model'):
        validate_bounded_walls([cube()])
    part=cube()
    part.update(open_surface=True, support_cutoff_mm=100, evidence_status='bounded_lidar_baseline')
    with pytest.raises(ValueError, match='omit current_model'):
        validate_bounded_walls([part])


def test_bounded_wall_baseline_is_accepted_as_a_prior_product():
    part=cube()
    part.update(open_surface=True,support_cutoff_mm=50,evidence_status='bounded_lidar_baseline')
    validate_bounded_walls([part])


def test_fresh_hypothesis_loses_unscanned_back_face(tmp_path):
    import laspy
    from run_architectural_flow import run
    # Only x=+0.1 was scanned. An opposite x=-0.1 wall skin must not
    # survive simply because the architectural candidate is a closed box.
    y,z=np.meshgrid(np.arange(-1,1.001,.015),np.arange(-1.5,1.501,.015))
    las=laspy.LasData(laspy.LasHeader(point_format=3,version='1.2'))
    las.header.scales=[.001,.001,.001]
    las.x=np.full(y.size,.1);las.y=y.ravel();las.z=z.ravel()
    scan_path=tmp_path/'one_side.las';las.write(scan_path)
    (tmp_path/'candidate.json').write_text(json.dumps({'parts':[cube()]}))
    manifest=tmp_path/'input.json'
    manifest.write_text(json.dumps({'label':'one-sided evidence regression','units':'metres',
        'candidate_model':'candidate.json','scans':[{'id':'scan','path':'one_side.las',
        'scan_to_model':np.eye(4).tolist()}]}))
    out=tmp_path/'result'
    report=run([manifest],out)
    result=json.loads((out/'model.build.json').read_text())['parts']
    assert len(result)==1
    v=np.asarray(result[0]['v'])
    assert v[:,0].min()>=.05-1e-7
    assert result[0]['open_surface']
    assert not report['site_accuracy_certified']
    assert (out/'model.glb').is_file()
    assert (out/'L0/review_top_front_side_3d.png').is_file()
