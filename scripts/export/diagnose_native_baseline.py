"""Investigate native edge splitting instead of concealing an area-audit failure."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely import union_all
from shapely.geometry import Polygon
from export_rectangular_skp import invoke

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'output_final/coverage_restored_v1/engrance'


def main():
    native=FOLDER/'Engrance_LiDAR_coverage_first.skp'
    target=FOLDER/'native_edge_diagnostic.json'
    audit=json.loads((FOLDER/'native_audit.json').read_text())
    names=[x['name'] for x in audit.get('errors',[])]
    if not names: return
    code=f'''
      m=Sketchup.active_model;
      raise 'Wrong model; do not switch unsaved sessions' unless m.path.gsub('\\\\','/').downcase=={json.dumps(native.as_posix().lower())};
      names={json.dumps(names)};
      rows=m.entities.grep(Sketchup::Group).select{{|g|names.include?(g.name)}}.map do |g|
        {{name:g.name,polygons:g.entities.grep(Sketchup::Face).map{{|f|f.outer_loop.vertices.map{{|v|v.position.to_a.map{{|x|x.to_f/39.37007874015748}}}}}}}}
      end;
      File.write({json.dumps(target.as_posix())},JSON.generate(rows));
      {{groups:rows.length}}.to_json
    '''
    print(invoke(code),flush=True)
    payload=json.loads((FOLDER/'model.build.json').read_text());parts={p['name']:p for p in payload['parts']}
    result=[]
    for row in json.loads(target.read_text()):
        part=parts[row['name']];v=np.array(part['v']);f=np.array(part['f'])
        def planes(polygons):
            groups=defaultdict(list)
            for t in polygons:
                t=np.asarray(t);axis=int(np.argmin(np.ptp(t,axis=0)))
                assert np.ptp(t[:,axis])<1e-8, 'Non-axis-aligned patch requires a different diagnostic'
                key=(axis,round(float(t[:,axis].mean()),7))
                groups[key].append(Polygon(t[:,[k for k in range(3) if k!=axis]]))
            return {k:union_all(vals) for k,vals in groups.items()}
        srcs=planes(v[f]);dsts=planes(row['polygons']);assert srcs.keys()==dsts.keys()
        native_vertices=np.vstack(row['polygons'])
        max_new_coordinate=float(cKDTree(v).query(native_vertices)[0].max())
        hausdorff=max(srcs[k].boundary.hausdorff_distance(dsts[k].boundary) for k in srcs)
        item={'name':row['name'],'source_union_area_m2':sum(g.area for g in srcs.values()),'native_union_area_m2':sum(g.area for g in dsts.values()),
              'added_area_mm2':sum(dsts[k].difference(srcs[k]).area for k in srcs)*1e6,'removed_area_mm2':sum(srcs[k].difference(dsts[k]).area for k in srcs)*1e6,
              'boundary_hausdorff_mm':hausdorff*1000,'max_new_vertex_distance_mm':max_new_coordinate*1000,
              'pass':hausdorff<=.000025 and max_new_coordinate<=1e-8}
        result.append(item)
    audit['edge_splitting_diagnostic']=result
    audit['native_geometry_verified']=all(x['pass'] for x in result)
    audit['verification_note']='Original strict area_consistency_pass preserved. Separate diagnostic permits <=0.025 mm boundary displacement only, and requires every native vertex to be an existing source coordinate.'
    (FOLDER/'native_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
