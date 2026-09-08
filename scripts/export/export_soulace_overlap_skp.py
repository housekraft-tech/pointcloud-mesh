"""Build, reopen and check the filtered native SketchUp handover."""
import json
from pathlib import Path
import numpy as np
import trimesh

from skp_client import rb
from audit_soulace_overlap import ROOT, say

OUT = ROOT/'output_final/soulace_overlap_only_50mm/release'
BUILD = OUT/'Soulace_overlap_only.build.json'
SKP = OUT/'Soulace_overlap_only.skp'


def run(code, timeout=1800):
    response = rb(code,timeout=timeout)
    if not response.get('ok'):
        raise RuntimeError(response)
    value = response['result']
    for _ in range(3):
        if not isinstance(value,str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value


def main():
    if not SKP.exists():
        ruby = ROOT/'scripts/export/ruby/pcm_overlap_surfaces.rb'
        say('Building native SketchUp open-surface model')
        result = run(f'load {json.dumps(ruby.as_posix())}; PCMOverlapSurfaces.build({json.dumps(BUILD.as_posix())}, {json.dumps(SKP.as_posix())})')
        say(result)
    native_audit = OUT/'native_geometry_audit.json'
    code = f'''
    raise 'Reopen failed' unless Sketchup.open_file({json.dumps(SKP.as_posix())});
    m = Sketchup.active_model;
    scale = 39.37007874015748;
    rows = m.entities.grep(Sketchup::Group).map do |g|
      faces = g.entities.grep(Sketchup::Face);
      {{name:g.name, faces:faces.length, area_m2:faces.sum{{|f| f.area}}/(scale*scale),
        edges:g.entities.grep(Sketchup::Edge).length, tag:g.layer.name}}
    end;
    File.write({json.dumps(native_audit.as_posix())},JSON.pretty_generate({{path:m.path,groups:rows,scenes:m.pages.map{{|p|p.name}}}}));
    {{path:m.path,groups:rows.length,faces:rows.sum{{|r|r[:faces]}}}}.to_json
    '''
    say(run(code))
    measured = json.loads(native_audit.read_text())
    planned = json.loads(BUILD.read_text())['parts']
    by_name = {p['name']:trimesh.Trimesh(p['v'],p['f'],process=False).area for p in planned}
    if set(by_name) != {g['name'] for g in measured['groups']}:
        raise ValueError('Native SketchUp group list differs from planned release')
    errors = []
    for row in measured['groups']:
        expected = float(by_name[row['name']])
        delta = row['area_m2']-expected
        row['expected_area_m2'] = expected
        row['area_delta_m2'] = delta
        # SketchUp may insert an existing nearly-collinear vertex onto a
        # triangle edge. The diagnosed L1 threshold changed area by 20.625 mm2
        # with only 0.0111 mm edge deviation; no new coordinates were created.
        if abs(delta) > max(.000025,expected*.00001):
            errors.append(row)
    measured['area_conservation_pass'] = not errors
    measured['absolute_area_tolerance_m2'] = .000025
    measured['relative_area_tolerance'] = .00001
    measured['native_edge_note'] = 'L1_floor_threshold_01: existing vertex inserted on a near-collinear edge; 20.625 mm2 area difference, 0.0111 mm maximum edge deviation. Native coordinate diagnostic retained.'
    measured['errors'] = errors
    native_audit.write_text(json.dumps(measured,indent=2))
    if errors:
        raise ValueError(f'{len(errors)} native groups failed area-conservation check')
    say('Native area consistency passed within documented import tolerances')
    for level in [0,1,2,None]:
        name = f'Soulace_L{level}_overlap_only_native.png' if level is not None else 'Soulace_overlap_only_native.png'
        png = OUT/name
        level_filter = f'tag.name.start_with?("OVERLAP_L{level}_") && ' if level is not None else ''
        code = f'''
        m=Sketchup.active_model;v=m.active_view;
        m.layers.each do |tag|
          next unless tag.name.start_with?('OVERLAP_');
          tag.visible = {level_filter}!tag.name.include?('CEILING');
        end;
        b=Geom::BoundingBox.new;
        m.entities.grep(Sketchup::Group).each{{|g|b.add(g.bounds) if g.layer.visible?}};
        c=b.center;s=[b.width,b.height,b.depth].max;
        fit=lambda do |eye,up|
          cam=Sketchup::Camera.new(eye,c,up,false);
          aspect=1900.0/1450.0;cam.aspect_ratio=aspect;
          right=cam.direction.cross(cam.up).normalize;
          vertical=cam.up.normalize;
          corners=(0..7).map{{|i|b.corner(i)-c}};
          xs=corners.map{{|p|p.dot(right)}};ys=corners.map{{|p|p.dot(vertical)}};
          cam.height=[ys.max-ys.min,(xs.max-xs.min)/aspect].max*1.12;
          v.camera=cam;
        end;
        fit.call(Geom::Point3d.new(c.x+1.3*s,c.y-1.5*s,c.z+1.2*s),Geom::Vector3d.new(0,0,1));
        m.rendering_options['EdgeDisplayMode']=0;
        m.rendering_options['DrawSilhouettes']=false;
        v.refresh;
        ok=v.write_image({{filename:{json.dumps(png.as_posix())},width:1900,height:1450,antialias:true,transparent:false}});
        raise 'Image failed' unless ok;
        {{image:{json.dumps(name)}}}.to_json
        '''
        if level is not None:
            code+=f'''
            [ ['L{level} - Top',Geom::Point3d.new(c.x,c.y,c.z+s),Geom::Vector3d.new(0,1,0)],
              ['L{level} - Front',Geom::Point3d.new(c.x,c.y-s,c.z),Geom::Vector3d.new(0,0,1)],
              ['L{level} - Side',Geom::Point3d.new(c.x+s,c.y,c.z),Geom::Vector3d.new(0,0,1)] ].each do |name,eye,up|
              fit.call(eye,up);
              page=m.pages[name] || m.pages.add(name);
              page.update;
            end;
            true
            '''
        say(run(code,timeout=600))
    say(run(f'm=Sketchup.active_model; m.pages[0].update; m.pages.selected_page=m.pages[0]; m.styles.update_selected_style if m.styles.respond_to?(:update_selected_style); m.save({json.dumps(SKP.as_posix())})'))
    measured['scenes']=run('Sketchup.active_model.pages.map{|p|p.name}.to_json')
    native_audit.write_text(json.dumps(measured,indent=2))
    say(f'Completed: {SKP}')


if __name__ == '__main__':
    main()
