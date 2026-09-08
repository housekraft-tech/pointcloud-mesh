"""Assemble the paired-face wall revision without discarding prior geometry."""
import json
from model_soulace_top_walls import ROOT,OUT,REC


def main():
    if list(OUT.glob('*.skp')):raise FileExistsError('Native revision exists')
    source=ROOT/'output_final/soulace_single_ground_plane_v3'
    old=json.loads((source/'native_audit.json').read_text())
    wall_audit=json.loads((OUT/'wall_fit_audit.json').read_text())
    replaced=['L2_'+w['name'] for w in wall_audit['walls'] if w['status']=='paired_face_wall_solid']
    parts=json.loads((OUT/'walls.build.json').read_text())['parts']
    roof=next(p for p in json.loads((REC/'recovered_parts.build.json').read_text())['parts'] if p['kind']=='roof_stair_reference')
    roof={**roof,'name':'Top-floor roof stair - crosswise scan reference','kind':'roof_stair_scan','level':2}
    # The roof scan corrects the omitted position/orientation, not walking-surface
    # interpretation. It remains observed evidence, not a fabricated stair solid.
    parts.append(roof)
    stairs=json.loads((ROOT/'output_final/soulace_clean_floor_stairs_v2/upper_stairs.build.json').read_text())['parts']
    scene_levels={p['name']:[1,2] for p in stairs}
    payload={'label':'Soulace | top-floor paired-face wall solids and crosswise roof-stair evidence',
             'source_native':str(source/'Soulace_one_ground_plane.skp'),'expected_base_groups':358,
             'reference_source_names':old['reference_only_groups']+replaced,
             'source_scene_levels':scene_levels,'parts':parts,
             'source_label':'Top walls: paired raw-LAS planes with modeled closure. Roof stair: full-density 50 mm bounded scan recovery.',
             'small_hole_interpolation':True,
             'note':'11 top-floor wall sections rebuilt as paired-face solids. Unpaired wall thickness unresolved; their earlier evidence retained. Crosswise roof stair is observed surface evidence, not a finished stair solid. Original geometry preserved as hidden reference when superseded. Single cyan ground plane unchanged. Not certified dimensions.'}
    (OUT/'additions.build.json').write_text(json.dumps(payload,separators=(',',':')))
    print({'modeled_wall_sections':len(replaced),'roof_reference_triangles':len(roof['f']),
           'base_groups':358,'expected_groups':358+len(parts)},flush=True)


if __name__=='__main__':main()
