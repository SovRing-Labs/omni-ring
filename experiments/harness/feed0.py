#!/usr/bin/env python3
"""FEED-0 (PREREGISTRATIONS-OMNITRI.md wave 4): the cube as a feeder in front of ring_headd, on 2,000 consecutive live notes.
A = today (one request per note). B = feeder: digest cache (exact text) + unique misses sent first, answers served from
the digest. Labels must be identical."""
import glob, json, os, socket, statistics, time

SOCK = "/tmp/ring_headd.sock"
HEAD = "d5.verb"


def ask(text):
    s = socket.socket(socket.AF_UNIX); s.settimeout(120); s.connect(SOCK)
    s.sendall(json.dumps({"head": HEAD, "text": text}).encode()); s.shutdown(socket.SHUT_WR); buf = b""
    while (k := s.recv(65536)):
        buf += k
    s.close(); return json.loads(buf)["label"]


def temp_c():
    best = None
    for z in glob.glob("/sys/class/thermal/thermal_zone*/"):
        try:
            if "x86_pkg_temp" in open(z + "type").read():
                best = int(open(z + "temp").read()) / 1000
        except OSError:
            pass
    return best


notes = []
for line in open("<your-data>/control-center/events.jsonl"):
    try:
        n = json.loads(line).get("note")
    except Exception:
        continue
    if n:
        notes.append(n)
notes = notes[-2000:]
res = {"notes": len(notes), "distinct": len(set(notes))}

time.sleep(15); t0c = temp_c()
lat, la = [], []
t = time.perf_counter()
for n in notes:
    a = time.perf_counter(); la.append(ask(n)); lat.append(time.perf_counter() - a)
wa = time.perf_counter() - t; ta = temp_c()
res["A_today"] = {"notes_per_s": len(notes) / wa, "p99_ms": sorted(lat)[int(0.99 * (len(lat) - 1))] * 1000,
                  "median_ms": statistics.median(lat) * 1000, "temp_start_c": t0c, "temp_end_c": ta, "wall_s": wa}

time.sleep(30); t0c = temp_c()
lat, lb, digest = [], [], {}
t = time.perf_counter()
for n in dict.fromkeys(notes):                            # unique misses first, in one burst
    digest[n] = ask(n)
for n in notes:
    a = time.perf_counter(); lb.append(digest[n]); lat.append(time.perf_counter() - a)
wb = time.perf_counter() - t; tb = temp_c()
res["B_feeder"] = {"notes_per_s": len(notes) / wb, "p99_ms_served": sorted(lat)[int(0.99 * (len(lat) - 1))] * 1000,
                   "temp_start_c": t0c, "temp_end_c": tb, "wall_s": wb, "embeds": len(digest)}
res["labels_identical"] = la == lb
res["speedup"] = res["B_feeder"]["notes_per_s"] / res["A_today"]["notes_per_s"]
res["verdict"] = {"PASS": res["speedup"] >= 1.5 and res["labels_identical"] and
                  (tb is None or ta is None or tb <= ta + 3)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.expanduser("~/gpurun-out-0930/feed0.result.json"), "w"), indent=1)
