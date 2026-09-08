"""Refresh generated-model metadata and produce the Engrance-labelled copy."""
import json
from pathlib import Path
from export_rectangular_skp import invoke

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output_final/rectangular_rebuild_v1'


def main():
    targets=[('soulace_l0','Soulace_L0_rectangular.skp'),
             ('mujammel','Koushik_Mujammel_rectangular.skp'),
             ('mujammel_strict','Engrance_strict_rectangular.skp'),
             ('soulace_entire_house','Soulace_entire_house_rectangular.skp')]
    ruby=ROOT/'scripts/export/ruby/pcm_rectangular.rb'
    for folder,name in targets:
        path=BASE/folder/name
        data=BASE/folder/'model.build.json'
        alias=BASE/folder/'Engrance_rectangular_continuity.skp' if folder=='mujammel' else None
        if alias and alias.exists():
            raise FileExistsError(alias)
        backup=BASE/'unsaved_session_before_metadata_refresh.skp'
        result=invoke(f'''
          m=Sketchup.active_model;
          if m.modified?
            raise 'Backup exists; preserve manually before continuing' if File.exist?({json.dumps(backup.as_posix())});
            raise 'Could not preserve active session' unless m.save_copy({json.dumps(backup.as_posix())});
          end;
          raise 'Could not reopen output' unless Sketchup.open_file({json.dumps(path.as_posix())});
          load {json.dumps(ruby.as_posix())};m=Sketchup.active_model;
          data=JSON.parse(File.read({json.dumps(data.as_posix())}));
          strict=PCMRectangular.stamp(m,data);
          m.pages.selected_page=m.pages[0];m.active_view.camera=m.pages[0].camera;
          m.rendering_options['DisplaySketchAxes']=false;
          m.pages[0].update;
          raise 'Save failed' unless m.save({json.dumps(path.as_posix())});
          {'raise "Alias save failed" unless m.save('+json.dumps(alias.as_posix())+');' if alias else ''}
          {{path:m.path,strict:strict,attrs:m.attribute_dictionary('Reconstruction').to_h}}.to_json
        ''')
        if not isinstance(result,dict):
            raise RuntimeError(result)
        (BASE/folder/'native_metadata.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
