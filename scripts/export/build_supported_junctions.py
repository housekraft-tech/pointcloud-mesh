"""Fit bounded, open junction faces; never extrude a guessed full plinth.

The fitting primitive is reusable. This case configuration is a declared scan
inspection ROI for the observed parking-facing foundation and porch sides.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import box
from filter_soulace_overlap import supported_polygon, mesh_from_polygon
from architectural_surface_refinement import proximity_evidence

ROOT = Path(__file__).resolve().parents[2]

def fit_bounded_vertical_face(points, name, normal_axis, seed_offset,
                              along_limits, z_limits, cutoff=.05):
    along_axis = 1 - normal_axis
    mask = ((np.abs(points[:, normal_axis]-seed_offset)<.025)
            & (points[:,along_axis]>=along_limits[0])
            & (points[:,along_axis]<=along_limits[1])
            & (points[:,2]>=max(z_limits[0],-.45))
            & (points[:,2]<=min(z_limits[1],-.12)))
    q=points[mask]
    if len(q)<500:
        raise ValueError('Insufficient observed junction face evidence')
    offset=float(np.median(q[:,normal_axis]))
    fit_error=np.abs(q[:,normal_axis]-offset)
    if np.percentile(fit_error,95)>.02:
        raise ValueError('The proposed junction is not a sufficiently planar surface')
    origin=np.zeros(3);origin[normal_axis]=offset
    u=np.zeros(3);u[along_axis]=1
    v=np.array([0,0,1.])
    tree=cKDTree(points)
    shape, bound=supported_polygon(box(along_limits[0],z_limits[0],along_limits[1],z_limits[1]),
                                  origin,u,v,tree,cutoff=cutoff)
    mesh=mesh_from_polygon(shape,origin,u,v)
    if mesh is None:
        raise ValueError('No junction surface survived evidence clipping')
    evidence=proximity_evidence(mesh.triangles,tree)
    if evidence['max_mm']>cutoff*1000+.001:
        raise ValueError('Junction evidence cutoff exceeded')
    part={'name':name,'kind':'plinth_observed_surface','level':0,
          'v':mesh.vertices.tolist(),'f':mesh.faces.tolist(),
          'colour':[166,161,148],'merge_coplanar_faces':True,'open_surface':True,
          'support_cutoff_mm':cutoff*1000,'evidence_status':'bounded_registered_scan_surface',
          'thickness_verified':False}
    report={'name':name,'fit_points':len(q),'normal_axis':normal_axis,'offset_m':offset,
            'fit_p95_mm':float(np.percentile(fit_error,95)*1000),
            'area_m2':float(mesh.area),'continuous_envelope':bound,'sampled_proximity':evidence,
            'bounds_m':mesh.bounds.tolist(),'solid_completion':False}
    return part,report

def main():
    out=ROOT/'output_final/architectural_flow_astra/parking_diagnostic'
    out.mkdir(parents=True,exist_ok=True)
    raw=np.load(ROOT/'output_final/soulace_clean_floor_stairs_v2/raw_detail.npy',mmap_mode='r')
    p=raw[(raw[:,0]>1.3)&(raw[:,0]<6.5)&(raw[:,1]>-4.2)&(raw[:,1]<1.7)&(raw[:,2]<.1)]
    parts=[];reports=[]
    for spec in [('Parking connection - observed long plinth face',0,1.718,[-3.9,.22],[-.5334,0]),
                 ('Parking connection - observed porch plinth face',1,.222,[1.718,5.17],[-.5334,0])]:
        part,report=fit_bounded_vertical_face(p,*spec)
        parts.append(part);reports.append(report)
    (out/'connection.build.json').write_text(json.dumps({'parts':parts}))
    (out/'connection.audit.json').write_text(json.dumps({'parts':reports,'site_accuracy_certified':False},indent=2))
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
