"""Talk to the bridge inside SketchUp: send Ruby, get the answer back."""
import json, socket, sys, time

def rb(code, port=9876, timeout=600):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.settimeout(timeout)
    s.sendall((json.dumps({"code": code}) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode(errors="replace"))

if __name__ == "__main__":
    r = rb(sys.stdin.read() if sys.argv[1:] == ["-"] else " ".join(sys.argv[1:]))
    print(json.dumps(r, indent=1)[:4000])
