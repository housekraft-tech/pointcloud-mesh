"""Read-only inventory of the final native Soulace model by level and kind."""
import json
from pathlib import Path
from collections import Counter
from export_rectangular_skp import invoke

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output_final/soulace_wall_floor_junctions_v7'
OUT=ROOT/'output_final/soulace_full_house_planes_v8'


def main():
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'source_native_inventory.json'
    code='''
      m=Sketchup.active_model;expected=SOURCE;
      raise 'Open the final top-floor source first' unless m.path.gsub('\\\\','/').downcase==expected.downcase;
      rows=m.entities.grep(Sketchup::Group).map do |g|
        {name:g.name,kind:g.get_attribute('CoverageBaseline','kind',''),level:g.get_attribute('CoverageBaseline','level',0),
         reference:g.get_attribute('CoverageBaseline','reference_only',false),hidden:g.hidden?,faces:g.entities.grep(Sketchup::Face).length,
         bounds_m:[g.bounds.min.to_a,g.bounds.max.to_a].map{|p|p.map{|c|c/PCMCoverageBaseline::SCALE}}}
      end;
      File.write(TARGET,JSON.pretty_generate({path:m.path,modified:m.modified?,groups:rows}));{groups:rows.length}.to_json
    '''
    source=SOURCE/'Soulace_complete_wall_planes_and_floor_junctions.skp'
    print(invoke(code.replace('SOURCE',json.dumps(source.as_posix())).replace('TARGET',json.dumps(target.as_posix()))),flush=True)
    data=json.loads(target.read_text());counts=Counter((r['level'],r['kind'],r['reference'],r['hidden']) for r in data['groups'])
    for key,count in sorted(counts.items(),key=lambda x:str(x[0])):print(key,count,flush=True)


if __name__=='__main__':main()
