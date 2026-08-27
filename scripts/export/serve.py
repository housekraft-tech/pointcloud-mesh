"""Serve the outputs, several requests at a time.

`python -m http.server` handles one request at a time. The viewers fetch GLBs of
70-80 MB, so a single one of those occupies the server for its whole download
and every other request -- a second viewer, the index, a texture -- waits behind
it and can time out into a blank page. That is not a bug in the pages.

This is the same static serving, threaded, with byte ranges so a browser can
resume or seek within a large mesh, and no caching so a rebuilt model is the one
you get on reload.
"""
import argparse, os, sys, threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # The pages are regenerated constantly and must never be stale. The
        # meshes are large and change rarely, so re-downloading and re-parsing
        # 80 MB on every reload is pure waste.
        big = self.path.endswith((".glb", ".ply", ".obj", ".npy", ".mp4"))
        self.send_header("Cache-Control",
                         "public, max-age=600" if big else "no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_head(self):
        """Serve byte ranges, so a large mesh can be resumed rather than restarted."""
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            size = os.path.getsize(path)
            first, _, last = rng[6:].partition("-")
            start = int(first) if first else 0
            end = int(last) if last else size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_error(416, "range not satisfiable")
                return None
            f = open(path, "rb")
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return _Ranged(f, end - start + 1)
        except OSError:
            self.send_error(404)
            return None

    def log_message(self, fmt, *args):
        if "404" in (args[1] if len(args) > 1 else "") or self.path.endswith(".html"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


class _Ranged:
    """A file that stops after N bytes, for a 206 response."""
    def __init__(self, f, n):
        self.f, self.left = f, n

    def read(self, n=-1):
        if self.left <= 0:
            return b""
        chunk = self.f.read(self.left if n < 0 else min(n, self.left))
        self.left -= len(chunk)
        return chunk

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    os.chdir(a.root)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port),
                              partial(Handler, directory=os.getcwd()))
    srv.daemon_threads = True
    print(f"serving {os.getcwd()} on http://127.0.0.1:{a.port} "
          f"({threading.active_count()} threads, ranges enabled)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
