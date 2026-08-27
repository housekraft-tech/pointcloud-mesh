# A socket into a running SketchUp, so it can be driven a command at a time.
#
# Same idea as sketchup-mcp: a TCP server inside SketchUp that takes Ruby and
# hands back the result. The reason to have it is debugging -- a batch script
# that runs at startup gives you one shot and a log file, and three attempts
# were lost to a Welcome window, a glob that matched nothing, and a lost run.
# With a socket the error comes straight back and the next command is a second
# away.
require 'sketchup.rb'
require 'socket'
require 'json'

module PCMBridge
  PORT = 9876
  @server = nil
  @clients = []

  LOG = File.join(ENV['TEMP'] || '.', 'pcm_bridge.log')

  def self.log(m)
    File.open(LOG, 'a') { |h| h.puts("#{Time.now.strftime('%H:%M:%S')} #{m}") }
  rescue
  end

  # Bind with SO_REUSEADDR and keep trying for a while.
  #
  # The port stays held for a minute or two after SketchUp is killed, so the
  # next launch could not bind -- and the failure was silent, which cost two
  # rounds of "why is nothing listening". Now it retries, and it writes down
  # what happened either way.
  def self.start
    return if @server
    attempt = 0
    timer = nil
    timer = UI.start_timer(2, true) do
      attempt += 1
      begin
        # SO_REUSEADDR has to be set BEFORE the bind, and TCPServer.new binds
        # immediately -- so setting it afterwards did nothing and every
        # relaunch after a kill hit EADDRINUSE until the port aged out. Build
        # the socket by hand instead.
        srv = Socket.new(:INET, :STREAM)
        srv.setsockopt(:SOCKET, :REUSEADDR, true)
        srv.bind(Addrinfo.tcp('127.0.0.1', PORT))
        srv.listen(8)
        @server = srv
        UI.stop_timer(timer)
        log("listening on #{PORT} after #{attempt} attempt(s)")
        UI.start_timer(0.2, true) { poll }
      rescue => e
        log("bind attempt #{attempt} failed: #{e.class}: #{e.message}")
        UI.stop_timer(timer) if attempt > 120
      end
    end
  end

  # Never run a command inside another one.
  #
  # SketchUp pumps window messages during a long operation -- a save of a
  # 2 MB model takes seconds -- so this 0.2 s timer fires WHILE an eval is
  # still running and starts a second one on top of it. Re-entering the Ruby
  # API like that took SketchUp down twice, both times on save.
  @busy = false

  def self.poll
    return if @busy
    @busy = true
    begin
      poll_once
    ensure
      @busy = false
    end
  end

  def self.poll_once
    begin
      while true
        pair = (@server.accept_nonblock(exception: false) rescue nil)
        break if pair.nil? || pair == :wait_readable
        @clients << pair[0]
      end
    rescue IO::WaitReadable, Errno::EAGAIN
    end
    @clients.each do |c|
      begin
        line = c.gets_nonblock rescue c.gets
        next if line.nil? || line.strip.empty?
        req = JSON.parse(line)
        out = { ok: true }
        begin
          val = eval(req['code'], TOPLEVEL_BINDING)
          out[:result] = val.inspect[0, 4000]
        rescue => e
          out = { ok: false, error: "#{e.class}: #{e.message}",
                  backtrace: (e.backtrace || [])[0, 3] }
        end
        c.puts(JSON.generate(out))
        c.flush
      rescue IO::WaitReadable, Errno::EAGAIN
      rescue => e
        @clients.delete(c) rescue nil
      end
    end
  end
end

PCMBridge.start
