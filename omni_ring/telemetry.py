"""
Telemetry on the torus (TORUS-TEL-1; J 2026-09-30: "turn our RAPL into a torus, with other system telemetry, used as the TLT
to feed the cpu"). Loop step: Mark (telemetry -> bus) -> Read -> Adjust -> Learn.

  Sensors        unprivileged machine readings: CPU use, mean clock (GHz), package temperature, temperature slope (EMA over
                 ~2 s: the 1 C sensor steps made a raw 0.5 s slope jump +-20 C/s). Package power (RAPL) only when readable:
                 root-only by default since PLATYPUS (CVE-2020-8694); the optional power plugin publishes a coarse reading.
  TorusEncoder   a generic RECORD encoder: any named scalars -> one torus frame. Each field is bound to its own role
                 codeword; its value is a fractional power (per-dimension frequency from SuperSeed), so nearby values give
                 nearby phases and distant values decorrelate (a random-Fourier-feature kernel on the torus R/Z). All
                 codewords come from SuperSeed v1, so frames are portable across dialects.
  publish        the frame goes on the omni-ring shm bus (ShmBridge "omniring_telemetry.bin"); readers: GearMemory, Novelty.
  GearMemory     the learned table: marks = past states, each with the gear that ran and the reward it earned. Read = sift the
                 current state against the marks, score each gear by its neighbours' rewards. Abstains when too few close
                 neighbours: the rule governor decides and the outcome becomes a new mark (that is how the table fills).
  Novelty        immune-tier reader: 1 - best similarity to known states. A state never seen before (hot while idle and
                 throttled = stuck fan / runaway process) scores high; ordinary idle and load do not.
Precedent (cited, not copied): Plate 1995 (HRR binding + clean-up), Rahimi & Recht 2007 (random Fourier features), Frady et
al. 2021 (fractional power encoding), TFHE programmable bootstrapping (a function read off a lookup table on the torus).
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import seed as S

K = S.K
PH = np.exp(2j * np.pi * np.arange(K) / K).astype(np.complex64)
BUS_NAME = "omniring_telemetry.bin"

# name, lo, hi (values are clipped into [lo, hi] and scaled to [0, 1])
MACHINE_FIELDS: List[Tuple[str, float, float]] = [
    ("util", 0.0, 1.0), ("ghz", 0.4, 4.2), ("temp", 40.0, 100.0), ("dtemp", -3.0, 3.0), ("watts", 0.0, 32.0),
]
POWER_STATE = "/run/omni-power/state.json"      # optional root helper (projects/omni-power): coarse watts, EPP requests
EPP_REQUEST = "/run/omni-power/epp_request"
BELT_FIELDS: List[Tuple[str, float, float]] = [("singles_waiting", 0.0, 32.0), ("batch_waiting", 0.0, 4000.0), ("p95_ms", 0.0, 200.0)]


class Sensors:
    """Unprivileged machine telemetry. read() returns util, ghz, temp, dtemp (+ watts when RAPL is readable)."""

    def __init__(self, slope_tau_s: float = 2.0):
        self.tau = slope_tau_s
        self._cpu = self._stat()
        self._t = time.perf_counter()
        self._temp = self.temp()
        self._slope = 0.0
        self._freqs = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))
        self._rapl = "/sys/class/powercap/intel-rapl:0/energy_uj"
        self._rapl_ok = os.access(self._rapl, os.R_OK)
        self._e = self._energy()

    @staticmethod
    def _stat() -> Tuple[int, int]:
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        return sum(v) - v[3] - v[4], sum(v)

    @staticmethod
    def temp() -> float:
        for z in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                if open(f"{z}/type").read().strip() == "x86_pkg_temp":
                    return int(open(f"{z}/temp").read()) / 1000
            except OSError:
                pass
        return float("nan")

    def _energy(self) -> Optional[int]:
        if not self._rapl_ok:
            return None
        try:
            return int(open(self._rapl).read())
        except OSError:
            return None

    def read(self) -> Dict[str, float]:
        now = time.perf_counter(); dt = max(now - self._t, 1e-3); self._t = now
        c = self._stat(); util = (c[0] - self._cpu[0]) / max(c[1] - self._cpu[1], 1); self._cpu = c
        t = self.temp()
        if t == t and self._temp == self._temp:
            a = min(1.0, dt / self.tau)
            self._slope += a * ((t - self._temp) / dt - self._slope)
        self._temp = t
        ghz = float(np.mean([int(open(p).read()) for p in self._freqs])) / 1e6 if self._freqs else float("nan")
        out = {"util": util, "ghz": ghz, "temp": t, "dtemp": self._slope}
        e = self._energy()
        if e is not None and self._e is not None and e >= self._e:
            out["watts"] = (e - self._e) / 1e6 / dt
        self._e = e
        if "watts" not in out:
            w = power_band()
            if w is not None:
                out["watts"] = w
        return out


def power_band(max_age_s: float = 5.0) -> Optional[float]:
    """Midpoint of the omni-power helper's current 2 W band, or None (no helper / stale reading)."""
    try:
        st = json.load(open(POWER_STATE))
        if time.time() - st["t"] <= max_age_s:
            return (st["watts_lo"] + st["watts_hi"]) / 2
    except (OSError, ValueError, KeyError):
        pass
    return None


def request_epp(value: Optional[str]) -> bool:
    """Ask the omni-power helper for an EPP value (it whitelists and expires requests after 30 s; None = restore)."""
    try:
        if value is None:
            os.remove(EPP_REQUEST)
        else:
            tmp = EPP_REQUEST + ".tmp"
            open(tmp, "w").write(value); os.replace(tmp, EPP_REQUEST)
        return True
    except OSError:
        return False


class TorusEncoder:
    """Named scalars -> one torus frame (complex, unit-ish length) and its 16-phase row for the sift."""

    def __init__(self, fields: Sequence[Tuple[str, float, float]], d: int = S.D, seed: int = 0, spread: float = 2.0):
        self.fields, self.d = list(fields), d
        # per-dimension frequencies in [-spread, spread] cycles over the field's range (SuperSeed nibbles, centred)
        self.freq = {n: (S.codeword("TEL:" + n, seed, d).astype(np.float32) - 7.5) / 7.5 * spread for n, _, _ in self.fields}
        self.role = {n: PH[S.codeword("R:" + n, seed, d)] for n, _, _ in self.fields}

    def encode(self, x: Dict[str, float]) -> np.ndarray:
        v = np.zeros(self.d, np.complex64); n = 0
        for name, lo, hi in self.fields:
            val = x.get(name)
            if val is None or val != val:
                continue                                   # a missing sensor contributes nothing (not a fake zero)
            u = float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
            v += self.role[name] * np.exp(2j * np.pi * self.freq[name] * u).astype(np.complex64)
            n += 1
        return v / np.sqrt(max(n, 1))

    @staticmethod
    def phases(v: np.ndarray) -> np.ndarray:
        return (np.round(np.angle(v) / (2 * np.pi / K)).astype(np.int64) % K).astype(np.uint8)


def publish(bridge, enc_phases: np.ndarray, confidence: float = 1.0) -> int:
    """Put a frame on the omni-ring shm bus (float32 phase angles, the ShmBridge layout)."""
    return bridge.write(enc_phases.astype(np.float32) * np.float32(2 * np.pi / K), [0, 0, 0, 0, 0], confidence)


def _self_score(d: int) -> float:
    return float(127 * 127 * d)


class GearMemory:
    """Learned shift table: state marks -> (gear, reward). Persisted as npz under ~/.cache/omni_ring/gear_memory/."""

    def __init__(self, n_gears: int, d: int = S.D, k: int = 12, min_sim: float = 0.55, min_votes: int = 3, cap: int = 4096,
                 path: Optional[str] = None):
        self.n_gears, self.d, self.k, self.min_sim, self.min_votes, self.cap = n_gears, d, k, min_sim, min_votes, cap
        self.path = path
        self.marks = np.zeros((0, d), np.uint8); self.gear = np.zeros(0, np.int16); self.reward = np.zeros(0, np.float32)
        if path and os.path.exists(path):
            z = np.load(path)
            if z["marks"].shape[1] == d:
                self.marks, self.gear, self.reward = z["marks"], z["gear"], z["reward"]

    def write(self, ph: np.ndarray, gear: int, reward: float) -> None:
        self.marks = np.vstack([self.marks, ph[None]])[-self.cap:]
        self.gear = np.append(self.gear, np.int16(gear))[-self.cap:]
        self.reward = np.append(self.reward, np.float32(reward))[-self.cap:]

    def save(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp.npz"
            np.savez(tmp, marks=self.marks, gear=self.gear, reward=self.reward); os.replace(tmp, self.path)

    def read(self, ph: np.ndarray) -> Tuple[Optional[int], Dict]:
        """-> (best gear or None to abstain, detail). Gear score = similarity-weighted mean reward of its close neighbours."""
        if len(self.marks) < self.min_votes:
            return None, {"why": "empty"}
        sc = S.sift_phase(ph, self.marks).astype(np.float64) / _self_score(self.d)
        top = np.argsort(-sc, kind="stable")[: self.k]
        top = top[sc[top] >= self.min_sim]
        if len(top) < self.min_votes:
            return None, {"why": "no close state", "best_sim": float(sc.max())}
        score, count = {}, {}
        for i in top:
            g = int(self.gear[i]); w = sc[i]
            score[g] = score.get(g, 0.0) + w * float(self.reward[i]); count[g] = count.get(g, 0.0) + w
        means = {g: score[g] / count[g] for g in score}
        best = max(means, key=lambda g: (means[g], -g))       # ties -> the lower (quieter) gear
        return best, {"means": {int(g): round(m, 3) for g, m in means.items()}, "n": int(len(top)), "best_sim": float(sc[top[0]])}


class Novelty:
    """Immune-tier reader: how unlike every known state is this frame? (0 = seen before, 1 = never)."""

    def __init__(self, d: int = S.D, cap: int = 4096):
        self.d, self.cap = d, cap
        self.known = np.zeros((0, d), np.uint8)

    def learn(self, ph: np.ndarray) -> None:
        self.known = np.vstack([self.known, ph[None]])[-self.cap:]

    def score(self, ph: np.ndarray) -> float:
        if not len(self.known):
            return 1.0
        return float(1.0 - S.sift_phase(ph, self.known).max() / _self_score(self.d))


def reward(p95_ms: float, served_per_s: float, temp: float, temp_target: float, p95_target_ms: float = 50.0) -> float:
    """Latency first, work second, heat as a penalty past the target. Unitless; higher is better."""
    heat = max(0.0, (temp - temp_target)) / 5.0 if temp == temp else 0.0
    return float(-p95_ms / p95_target_ms + np.log1p(served_per_s) / 5.0 - heat)
