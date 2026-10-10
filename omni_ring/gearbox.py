"""
Gearbox — an automatic transmission for the belt (J 2026-09-30: "tune for high rpm, rev and switch to higher throughput when
threadlocking"; lineage BELT-0 + sweep). Gears are the measured belt settings; a governor thread shifts between them from live
telemetry, with hysteresis so it never hunts.

Telemetry per window (default 0.5 s): single backlog, batch backlog (queries), whole-CPU load, package temperature, recent
single p95 latency. Shift rules:
  UP    backlog grew for `up_after` consecutive windows, or load >= `lock_load` with backlog ("threadlock"): next gear up
  DOWN  no backlog and load < `idle_load` for `down_after` windows: next gear down
  HEAT  package temperature >= temp_cap: shift to the quiet gear and hold until `temp_cap - 5`
  GUARD overdrive (whole-job batching) starves singles, so it is left the moment any single waits
Every window is appended to `telemetry` (the marks TUNE-0 will learn shift points from).

PROFILES (J 2026-09-30: "thermal management on legacy/underpowered edge devices, transmission on capable builds"):
  thermal       heat-first. Starts in quiet; temperature is the primary input: shift DOWN whenever temp >= temp_target,
                shift UP (at most to `thermal_max_gear`) only with >= 8 C of headroom AND backlog; calm returns to quiet.
                GEAR-0 on the i5-8365U (15 W) reached 98-100 C in every arm and quiet (4 + 4 lanes) won under sustained load.
  transmission  queue-first (the rules above), starts in cruise; heat is an emergency cap with a TIMED hold (<= heat_hold_s),
                never a permanent lock (GEAR-0's stuck hold).
  auto          detect_profile(): a one-off 3 s full-core probe (temperature rise and peak), cached per machine in
                ~/.cache/omni_ring/device_profile.json. RAPL limits are not used: firmware reports placeholders (200 W long-term
                on a 15 W chip).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from .belt import Belt

# Gears, low to high. quiet = steadiest hum (sweep: load sd 0.11); cruise = the sweep winner (ss ss ss ss B);
# power = same rhythm with bigger chunks; overdrive = whole-job batching at full threads (max throughput, singles wait).
GEARS: List[Dict] = [
    {"name": "quiet", "mode": "express", "express_threads": 4, "threads": 8},
    {"name": "cruise", "mode": "rhythm", "ratio": 4, "single_threads": 2, "chunk": 32, "threads": 8},
    {"name": "power", "mode": "rhythm", "ratio": 4, "single_threads": 2, "chunk": 96, "threads": 8},
    {"name": "overdrive", "mode": "fifo", "threads": 8},
]


# step 8: gear -> CPU energy/performance preference (only with the omni-power helper)
EPP_BY_GEAR = ["balance_power", "balance_performance", "performance", "performance"]


def _pkg_temp() -> float:
    import glob
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(f"{z}/type").read().strip() == "x86_pkg_temp":
                return int(open(f"{z}/temp").read()) / 1000
        except OSError:
            pass
    return float("nan")


def detect_profile(probe_s: float = 3.0, cache: bool = True, refresh: bool = False) -> Dict:
    """Feel the engine: run a short full-core sift load and watch the package temperature. Thermal-limited -> 'thermal'."""
    import json, os
    path = os.path.expanduser("~/.cache/omni_ring/device_profile.json")
    if cache and not refresh and os.path.exists(path):
        return json.load(open(path))
    from . import seed as S
    from .belt import _threads
    lat = np.stack([S.codeword(f"probe{i}") for i in range(2048)])
    qs = lat[:64]
    t_start, t0 = _pkg_temp(), time.perf_counter()
    temps = []
    _threads(os.cpu_count() or 1)
    while time.perf_counter() - t0 < probe_s:
        S.sift_batch(qs, lat)
        temps.append((time.perf_counter() - t0, _pkg_temp()))
    _threads(1)
    peak = max(t for _, t in temps) if temps else float("nan")
    slope = (peak - t_start) / probe_s if t_start == t_start else float("nan")
    thermal = not (peak == peak) or peak >= 90 or slope >= 4.0
    out = {"profile": "thermal" if thermal else "transmission", "start_c": t_start, "peak_c": peak,
           "slope_c_per_s": round(slope, 2), "cpus": os.cpu_count(), "probe_s": probe_s,
           "rule": "thermal if peak >= 90 C or rise >= 4 C/s (or no sensor)"}
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(out, open(path, "w"), indent=1)
    return out


def _cpu():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return sum(v) - v[3] - v[4], sum(v)


class Gearbox:
    def __init__(self, belt: Belt, start_gear: Optional[int] = None, window: float = 0.5, up_after: int = 2, down_after: int = 4,
                 lock_load: float = 0.92, idle_load: float = 0.5, temp_cap: float = 95.0, gears: Optional[List[Dict]] = None,
                 profile: str = "auto", temp_target: float = 90.0, thermal_max_gear: int = 1, heat_hold_s: float = 10.0,
                 governor: str = "rules", memory_path: Optional[str] = None, torus_margin: float = 0.05, bus: bool = False,
                 epp: bool = False):
        self.belt, self.gears, self.window = belt, gears or GEARS, window
        self.device = detect_profile() if profile == "auto" else {"profile": profile}
        self.profile = self.device["profile"]
        self.temp_target, self.thermal_max_gear, self.heat_hold_s = temp_target, thermal_max_gear, heat_hold_s
        self.gear = start_gear if start_gear is not None else (0 if self.profile == "thermal" else 1)
        self.up_after, self.down_after, self.lock_load, self.idle_load, self.temp_cap = up_after, down_after, lock_load, idle_load, temp_cap
        self.telemetry: List[Dict] = []
        self.shifts: List[tuple] = []
        self._run = False
        self._hot = False
        # governor: "rules" (GEAR-0/1) or "torus" (TORUS-TEL-1: learned shift table on the torus, rules as safety + explorer)
        assert governor in ("rules", "torus")
        self.governor, self.torus_margin, self._prev, self.bus = governor, torus_margin, None, None
        # epp (step 8, optional root helper projects/omni-power): each gear also asks for a CPU energy/performance preference
        self.epp = epp and os.path.isdir("/run/omni-power")
        if governor == "torus":
            from . import telemetry as TT
            self.sensors = TT.Sensors()
            self.enc = TT.TorusEncoder(TT.MACHINE_FIELDS + TT.BELT_FIELDS)
            path = memory_path or os.path.expanduser(f"~/.cache/omni_ring/gear_memory/{self.profile}.npz")
            self.memory = TT.GearMemory(len(self.gears), path=path)
            self.novelty = TT.Novelty()
            if bus:
                from .shm_bridge import ShmBridge
                self.bus = ShmBridge(name=TT.BUS_NAME)

    def _apply(self, g: int, why: str) -> None:
        if g == self.gear:
            return
        cfg = {k: v for k, v in self.gears[g].items() if k != "name"}
        self.belt.reconfigure(**cfg)
        self.shifts.append((round(time.perf_counter() - self.t0, 2), self.gears[self.gear]["name"], self.gears[g]["name"], why))
        self.gear = g

    def start(self) -> "Gearbox":
        cfg = {k: v for k, v in self.gears[self.gear].items() if k != "name"}
        for k, v in cfg.items():
            setattr(self.belt, k, v)
        self.belt.start()
        self._run, self.t0 = True, time.perf_counter()
        self.th = threading.Thread(target=self._govern, daemon=True)
        self.th.start()
        return self

    def stop(self) -> None:
        self._run = False
        self.th.join(timeout=5)
        self.belt.stop()

    # -- telemetry window ------------------------------------------------------------------------------------------
    def _window(self, a):
        b = _cpu(); load = (b[0] - a[0]) / max(b[1] - a[1], 1)
        s_wait, b_wait = self.belt.backlog()
        lat = list(self.belt.latencies)[-64:]
        p95 = float(np.percentile(lat, 95)) * 1000 if lat else 0.0
        tel = {"t": round(time.perf_counter() - self.t0, 2), "gear": self.gears[self.gear]["name"], "load": round(load, 3),
               "singles_waiting": s_wait, "batch_waiting": b_wait, "temp_c": _pkg_temp(), "single_p95_ms": round(p95, 1),
               "served": dict(self.belt.served)}
        return tel, b

    # -- rule governors (one step each; st carries their counters) ---------------------------------------------------
    def _rule_transmission(self, st: Dict, tel: Dict):
        """-> (gear, why) or None. Queue-first; heat is an emergency cap with a timed hold."""
        load, s_wait, temp = tel["load"], tel["singles_waiting"], tel["temp_c"]
        backlog = s_wait + tel["batch_waiting"]
        now = time.perf_counter()
        if temp == temp and temp >= self.temp_cap and st["hot_since"] is None:
            st["hot_since"] = now
            return 0, f"heat {temp:.0f}C"
        if st["hot_since"] is not None:
            if temp < self.temp_cap - 5 or now - st["hot_since"] >= self.heat_hold_s:
                st["hot_since"] = None
            else:
                return self.gear, "heat hold"
        if self.gears[self.gear]["mode"] == "fifo" and s_wait > 0:
            st["grow"] = st["calm"] = 0
            return self.gear - 1, "single waiting in overdrive"
        st["grow"] = st["grow"] + 1 if backlog > st["prev_backlog"] and backlog > 0 else 0
        locked = load >= self.lock_load and backlog > 0
        st["calm"] = st["calm"] + 1 if backlog == 0 and load < self.idle_load else 0
        st["prev_backlog"] = backlog
        top = len(self.gears) - 1
        if (st["grow"] >= self.up_after or locked) and self.gear < top:
            nxt = self.gear + 1
            if self.gears[nxt]["mode"] == "fifo" and s_wait > 0:
                nxt = self.gear if self.gear >= 2 else self.gear + 1   # never shift INTO overdrive over waiting singles
            if nxt != self.gear:
                st["grow"] = 0
                return nxt, "threadlock" if locked else "backlog growing"
        elif st["calm"] >= self.down_after and self.gear > 1:
            st["calm"] = 0
            return self.gear - 1, "calm"
        return None

    def _rule_thermal(self, st: Dict, tel: Dict):
        """-> (gear, why) or None. Heat-first shifting for thermally limited devices (see PROFILES)."""
        load, temp = tel["load"], tel["temp_c"]
        backlog = tel["singles_waiting"] + tel["batch_waiting"]
        st["grow"] = st["grow"] + 1 if backlog > st["prev_backlog"] and backlog > 0 else 0
        st["calm"] = st["calm"] + 1 if backlog == 0 and load < self.idle_load else 0
        st["prev_backlog"] = backlog
        if temp == temp and temp >= self.temp_target and self.gear > 0:
            st["grow"] = 0
            return self.gear - 1, f"at temp target {temp:.0f}C"
        headroom = temp != temp or temp <= self.temp_target - 8
        if headroom and (st["grow"] >= self.up_after or (load >= self.lock_load and backlog > 0)) and self.gear < self.thermal_max_gear:
            st["grow"] = 0
            return self.gear + 1, f"backlog with headroom ({temp:.0f}C)"
        if st["calm"] >= self.down_after and self.gear > 0:
            st["calm"] = 0
            return 0, "calm"
        return None

    # -- torus governor (TORUS-TEL-1): the learned table proposes, the rules keep safety and explore -----------------
    def _torus_step(self, tel: Dict, rule):
        from . import telemetry as TT
        x = self.sensors.read()
        x.update(singles_waiting=tel["singles_waiting"], batch_waiting=tel["batch_waiting"], p95_ms=tel["single_p95_ms"])
        ph = self.enc.phases(self.enc.encode(x))
        if self.bus is not None:
            TT.publish(self.bus, ph)
        now = time.perf_counter()
        served = sum(tel["served"].values())
        if self._prev is not None:                           # learn: last window's state + gear earned this reward
            pph, pg, pserved, pt = self._prev
            r = TT.reward(tel["single_p95_ms"], (served - pserved) / max(now - pt, 1e-3), x["temp"], self.temp_target)
            self.memory.write(pph, pg, r); self.novelty.learn(pph)
        nov = self.novelty.score(ph)
        self._prev = (ph, self.gear, served, now)
        tel.update(novelty=round(nov, 3), ghz=round(x["ghz"], 2), dtemp=round(x["dtemp"], 2))
        if "watts" in x:
            tel["watts"] = round(x["watts"], 2)
        if rule is not None and ("heat" in rule[1] or "temp target" in rule[1] or "overdrive" in rule[1]):
            tel["decider"] = "rules:safety"
            return rule                                     # safety always wins over the table
        g, info = self.memory.read(ph)
        if g is None:
            tel["decider"] = "rules:" + info["why"]
            return rule                                     # abstain -> the rules decide (and the outcome becomes a mark)
        tel["decider"] = f"torus {info['means']}"
        cur = info["means"].get(self.gear)
        if g == self.gear or (cur is not None and info["means"][g] - cur < self.torus_margin):
            return None
        if self.profile == "thermal" and g > self.thermal_max_gear:
            return None
        nxt = self.gear + (1 if g > self.gear else -1)      # one gear at a time, never a jump
        if self.gears[nxt]["mode"] == "fifo" and tel["singles_waiting"] > 0:
            return None
        return nxt, f"torus table (n={info['n']}, sim={info['best_sim']:.2f})"

    def _govern(self) -> None:
        st = {"grow": 0, "calm": 0, "prev_backlog": 0, "hot_since": None}
        rules = self._rule_thermal if self.profile == "thermal" else self._rule_transmission
        a = _cpu()
        while self._run:
            time.sleep(self.window)
            tel, a = self._window(a)
            prop = rules(st, tel)
            if self.governor == "torus":
                prop = self._torus_step(tel, prop)
            self.telemetry.append(tel)
            if prop is not None and prop[0] != self.gear:          # a hold returns the current gear: no shift
                self._apply(prop[0], prop[1])
            if self.epp:                                            # refresh every window (the helper expires requests)
                from . import telemetry as TT
                TT.request_epp(EPP_BY_GEAR[min(self.gear, len(EPP_BY_GEAR) - 1)])
        if self.governor == "torus":
            self.memory.save()
        if self.epp:
            from . import telemetry as TT
            TT.request_epp(None)                                    # hand the machine its own preference back
