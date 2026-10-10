#!/usr/bin/env python3
"""IPC-0 (PREREGISTRATIONS-OMNITRI.md wave 4): round trip of a 1 KB request through (a) a spawned Python process, (b) a resident
Python server over a Unix socket, (c) a shared-memory ring with a resident C worker; plus today's real path
(one bin/fleet-exec read op as a subprocess)."""
import json, mmap, os, socket, statistics, struct, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
PAYLOAD = bytes(range(256)) * 4                           # 1 KB
res = {}


def stats(ts):
    ts = sorted(ts)
    return {"median_us": statistics.median(ts) * 1e6, "p99_us": ts[int(0.99 * (len(ts) - 1))] * 1e6, "n": len(ts)}


# (a) spawn a Python process per request
ECHO = "import sys; d=sys.stdin.buffer.read(); sys.stdout.buffer.write(d[::-1])"
ts = []
for _ in range(200):
    t = time.perf_counter(); out = subprocess.run([sys.executable, "-c", ECHO], input=PAYLOAD, capture_output=True).stdout
    ts.append(time.perf_counter() - t); assert out == PAYLOAD[::-1]
res["a_spawn_python"] = stats(ts)

# (b) resident Python server over a Unix socket
SOCK = "/tmp/ipc0.sock"
if os.path.exists(SOCK):
    os.unlink(SOCK)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(SOCK); srv.listen(1)


def serve():
    c, _ = srv.accept()
    while True:
        d = b""
        while len(d) < 1024:
            k = c.recv(1024 - len(d))
            if not k:
                return
            d += k
        c.sendall(d[::-1])


threading.Thread(target=serve, daemon=True).start()
cl = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); cl.connect(SOCK)
ts = []
for _ in range(10000):
    t = time.perf_counter(); cl.sendall(PAYLOAD); d = b""
    while len(d) < 1024:
        d += cl.recv(1024 - len(d))
    ts.append(time.perf_counter() - t)
assert d == PAYLOAD[::-1]
res["b_socket_resident_python"] = stats(ts)
cl.close()

# (c) shared-memory ring + resident C worker
subprocess.run(["gcc", "-O2", "-march=native", "-o", f"{HERE}/c/ipc_worker", f"{HERE}/c/ipc_worker.c", "-lrt"], check=True)
fd = os.open("/dev/shm/ipc0", os.O_CREAT | os.O_RDWR, 0o600); os.ftruncate(fd, 4096)
m = mmap.mmap(fd, 4096); m[:16] = b"\0" * 16
worker = subprocess.Popen([f"{HERE}/c/ipc_worker"]); time.sleep(0.2)
ts = []
for i in range(1, 10001):
    t = time.perf_counter()
    m[64:1088] = PAYLOAD
    m[0:8] = struct.pack("<Q", i)                          # publish request (x86: stores are ordered)
    while struct.unpack_from("<Q", m, 8)[0] != i:
        pass
    reply = m[2048:3072]
    ts.append(time.perf_counter() - t)
assert reply == PAYLOAD[::-1]
res["c_shm_ring_c_worker"] = stats(ts)
m[64:68] = b"QUIT"; m[0:8] = struct.pack("<Q", 10_001); worker.wait(timeout=5)
m.close(); os.close(fd); os.unlink("/dev/shm/ipc0")

# today's real path: one fleet-exec read op as a subprocess
FE = "<your-data>/bin/fleet-exec"
op = json.dumps({"operations": [{"type": "read", "path": "<your-data>/README.md"}]})
ts = []
for _ in range(20):
    t = time.perf_counter(); subprocess.run([FE, op], capture_output=True, env=dict(os.environ, FLEET_EXEC_CLASSIFIER="off"))
    ts.append(time.perf_counter() - t)
res["today_fleet_exec_subprocess"] = stats(ts)

for k, v in res.items():
    print(f"IPC-0 {k:30s} median {v['median_us']:>12,.1f} us  p99 {v['p99_us']:>12,.1f} us  (n={v['n']})", flush=True)
c, a = res["c_shm_ring_c_worker"]["median_us"], res["a_spawn_python"]["median_us"]
res["verdict"] = {"PASS": c <= 50 and a / c >= 1000, "spawn_over_shm": a / c,
                  "fleet_exec_over_shm": res["today_fleet_exec_subprocess"]["median_us"] / c}
print("IPC-0:", res["verdict"], flush=True)
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/ipc0.result.json"), "w"), indent=1)
