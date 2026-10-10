#!/usr/bin/env python3
"""IPC-1 equivalence: 20 operations through bin/fleet-exec (subprocess) and bin/fleet-execd (fork server) must give the
same JSON after removing timing/pid/log fields."""
import json, os, socket, subprocess, sys, tempfile

AI = "<your-workspace>"; FE = f"{AI}/bin/fleet-exec"; SOCK = "/tmp/fleet-execd.sock"
DROP = {"duration_ms", "pid", "log_path", "ts", "timestamp", "elapsed", "elapsed_ms", "_capsule", "started_at", "wall_ms",
        "_area_warnings", "mtime", "_intent"}
tmp = tempfile.mkdtemp(dir="/tmp")


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def via_sub(argv):
    r = subprocess.run([FE, *argv], capture_output=True, text=True, cwd=tmp, env={**os.environ, "FLEET_EXEC_CLASSIFIER": "off"})
    return r.returncode, r.stdout


def via_daemon(argv):
    s = socket.socket(socket.AF_UNIX); s.connect(SOCK)
    s.sendall((json.dumps({"argv": argv, "cwd": tmp, "env": {"FLEET_EXEC_CLASSIFIER": "off"}}) + "\n").encode()); b = b""
    while (k := s.recv(65536)):
        b += k
    r = json.loads(b); return r["code"], r["stdout"]


P = lambda ops, **kw: json.dumps({"operations": ops, **kw})  # noqa: E731
cases = [
    [P([{"type": "read", "path": f"{AI}/CLAUDE.md"}])],
    [P([{"type": "bash", "command": "echo hi"}])],
    [P([{"type": "bash", "command": "ls <your-data>/bin | head -3"}])],
    [P([{"type": "glob", "pattern": "*.py", "path": f"{AI}/experiments/harness"}])],
    [P([{"type": "grep", "pattern": "def tile_shape", "path": f"{AI}/packages/omni-ring/omni_ring"}])],
    [P([{"type": "sql", "database": "dispatch", "query": "select count(*) from tasks"}])],
    [P([{"type": "sql", "database": "dispatch", "query": "create table x(a)"}])],          # refused without mode
    [P([{"type": "write", "path": f"{tmp}/a.txt", "content": "alpha"}])],
    [P([{"type": "write", "path": f"{tmp}/b.txt", "content": "b"}, {"type": "read", "path": f"{tmp}/b.txt"}], concurrency="auto")],
    [P([{"type": "bash", "command": "rm -rf /"}])],                                         # guard block
    [P([{"type": "bash", "command": "git push --force origin main"}])],                     # guard block
    ["not json at all"],                                                                    # malformed
    [P([{"type": "nope"}])],                                                                # invalid op type
    ["--validate-only", P([{"type": "bash", "command": "true"}])],
    ["--validate-only", P([{"type": "nope"}])],
    [P([{"type": "bash", "command": "echo 'unterminated"}])],                               # doctor repair path
    [P([{"type": "bash", "command": "exit 3"}])],
    [P([{"type": "read", "path": "/nonexistent/file"}])],
    [P([{"type": "bash", "command": "printf '%s' x"}, {"type": "bash", "command": "echo y"}])],
    [json.dumps([{"type": "bash", "command": "echo list-form"}])],                          # bare list payload
]
same = 0; diffs = []
for i, argv in enumerate(cases):
    ca, oa = via_sub(argv); cb, ob = via_daemon(argv)
    try:
        ja, jb = strip(json.loads(oa)), strip(json.loads(ob))
    except Exception:
        ja, jb = oa.strip(), ob.strip()
    ok = ca == cb and ja == jb
    same += ok
    if not ok:
        diffs.append({"case": i, "argv": argv[-1][:80], "sub": [ca, str(ja)[:300]], "daemon": [cb, str(jb)[:300]]})
print(f"IPC-1 equivalence: {same}/{len(cases)} identical")
for d in diffs:
    print(json.dumps(d)[:700])
json.dump({"identical": same, "n": len(cases), "diffs": diffs}, open(os.path.expanduser("~/gpurun-out-0930/ipc1_equiv.json"), "w"), indent=1)
