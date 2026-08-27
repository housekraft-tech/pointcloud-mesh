# Turn every .dae in a folder into a real .skp, then quit.
#
# There is no open writer for .skp -- the format is closed and only SketchUp's
# own SDK can produce one. But SketchUp is ON this machine, and its Ruby API
# can be told to import and save. So the conversion runs inside SketchUp
# itself: this file is dropped in the Plugins folder, SketchUp is launched, and
# it works through the queue and exits.
#
# The folder and the log are passed through environment variables so the same
# script serves every batch.
require 'sketchup.rb'
require 'fileutils'

module PointcloudMesh
  def self.log(f, msg)
    File.open(f, 'a') { |h| h.puts("#{Time.now.strftime('%H:%M:%S')}  #{msg}") }
  end

  def self.convert_all
    dir = ENV['PCM_DAE_DIR']
    logf = ENV['PCM_LOG'] || File.join(dir.to_s, 'skp_convert.log')
    return if dir.nil? || dir.empty?
    log(logf, "started in #{dir}")
    # Dir.glob reads a backslash as an escape, so a Windows path matches
    # nothing at all -- the first run reported "0 .dae files" in a folder with
    # eight of them. Forward slashes work on Windows and keep glob honest.
    pattern = File.join(dir.tr('\', '/'), '**', '*.dae')
    files = Dir.glob(pattern).sort
    log(logf, "pattern #{pattern}")
    log(logf, "#{files.length} .dae files")
    files.each do |dae|
      skp = dae.sub(/\.dae\z/i, '.skp')
      if File.exist?(skp) && File.mtime(skp) > File.mtime(dae)
        log(logf, "skip #{File.basename(skp)} (already newer)")
        next
      end
      begin
        Sketchup.file_new                       # one model per file, nothing carried over
        model = Sketchup.active_model
        model.options['UnitsOptions']['LengthUnit'] = 2   # metres
        ok = model.import(dae, false)
        ents = model.entities.length
        # SketchUp imports Collada into a component; explode it once so the
        # parts land as top-level groups the designer can click, not one nested
        # instance they have to open first.
        model.entities.grep(Sketchup::ComponentInstance).each { |i| i.explode }
        model.save(skp)
        log(logf, "#{ok ? 'ok  ' : 'FAIL'} #{File.basename(dae)} -> " \
                  "#{File.basename(skp)}  (#{ents} entities, " \
                  "#{(File.size(skp)/1024.0).round} kB)")
      rescue => e
        log(logf, "ERROR #{File.basename(dae)}: #{e.message}")
      end
    end
    log(logf, 'DONE')
    Sketchup.quit if ENV['PCM_QUIT'] == '1'
  end
end

# Wait until there really is a model to work in. A one-shot timer fired too
# early on the first run -- SketchUp was still on its Welcome window, there was
# no active model, and the whole batch silently did nothing.
PointcloudMesh.instance_variable_set(:@tries, 0)
id = nil
id = UI.start_timer(2, true) do
  n = PointcloudMesh.instance_variable_get(:@tries) + 1
  PointcloudMesh.instance_variable_set(:@tries, n)
  if Sketchup.active_model
    UI.stop_timer(id)
    PointcloudMesh.convert_all
  elsif n > 30
    UI.stop_timer(id)
  end
end
