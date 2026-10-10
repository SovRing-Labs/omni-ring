"""PLM-1 gap detector and PLM-3 style features / memory."""
import numpy as np
from omni_ring.gap import GapDetector
from omni_ring.style import StyleMemory, features, reward


def test_gap_declared_on_abstain_rate_not_single_scores():
    rng = np.random.default_rng(0)
    good = [{"head": "d5.verb", "item": f"e{i}", "pred": "STEP", "margin": float(rng.uniform(0.5, 1))} for i in range(40)]
    weak = [{"head": "d4.error_class", "item": f"o{i}", "pred": "other", "margin": float(m),
             "because": [["d4_toolfix:12", "other", 0.3]]} for i, m in enumerate(rng.uniform(0, 0.4, 40))]
    few = [{"head": "d1.role", "item": "t1", "pred": "dev", "margin": 0.0}]          # below min_n: never a gap
    gaps = GapDetector(default_tau=0.2).scan(good + weak + few)
    assert [g.head for g in gaps] == ["d4.error_class"]
    b = GapDetector().brief(gaps[0])
    assert "VERIFY" in b and "d4.error_class" in b and "d4_toolfix" in b


def test_unproved_claims_raise_a_gap():
    recs = [{"head": "d2.outcome", "item": f"d{i}", "pred": "done", "margin": 0.9} for i in range(25)]
    gaps = GapDetector().scan(recs, unproved={"d2.outcome": 3})
    assert gaps and gaps[0].unproved == 3


def test_style_features_separate_tables_from_prose():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n- item one\n- item two"
    prose = ("The ring holds every mark we write. It answers by resonance, and when nothing resonates it says so. "
             "Short. Then a longer sentence that carries the melody and resolves what the short one opened.")
    ft, fp = features(table), features(prose)
    assert ft["list_density"] > 0.8 > fp["list_density"]
    assert fp["sent_cv"] > 0.3 and 0 < fp["sonority"] < 1


def test_style_memory_learns_context_to_preset():
    m = StyleMemory(["dense-table", "conversational"])
    report = features("| x | y |\n|---|---|\n| 1 | 2 |\n- a\n- b")
    musing = features("What if the torus were a cortex? Then every spark would have a place, and a place a meaning.")
    for _ in range(6):
        m.write(report, "dense-table", 1.0); m.write(report, "conversational", 0.2)
        m.write(musing, "conversational", 1.0); m.write(musing, "dense-table", 0.3)
    assert m.choose(report) == "dense-table" and m.choose(musing) == "conversational"
    assert reward("abc", None) == 1.0 and reward("abcdef", "abcxyz", follow_up=True) < 0.5


def test_prosody_measures_landing_rest_and_ring():
    from omni_ring.style import prosody
    t = ("Evening water climbs the patient shore through a long hour of gathering light, gathering weight, rising over "
         "stones that remember every earlier tide and every quiet return. Then it falls back. The water does not rest; "
         "it returns, rather than ending, to the patient shore where evening water began.")
    p = prosody(t)
    assert p["rests"] > 0 and p["antithesis"] > 0 and p["ring"] > 0 and 0 <= p["landing"] <= 1
