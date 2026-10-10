"""CORTEX-2 step 1 gate G1 (pre-registered in CORTEX-2 plan (unpublished) before this file existed)."""
import numpy as np
import pytest

from omni_ring import seed
from omni_ring.vm import VM, ProgramCodebook, iid, L


@pytest.fixture(scope="module")
def world():
    labels = [f"golden-{i}" for i in range(16)] + ["", "ümlaut", "E:Paris", "R:P36"]
    lattice = np.stack([seed.codeword(s, 0) for s in labels])          # 20 x D uint8
    cb = ProgramCodebook(lattice.shape[1])
    return lattice, cb


def test_sift_program_matches_kernel(world):
    lattice, cb = world
    vm = VM(lattice, cb)
    vm.load_program(cb.write([iid("LOAD", 0), iid("SIFT", 7), iid("THRESHOLD", 4)]))
    for i in range(16):
        vm.set(0, lattice[i])
        m = vm.tick()
        assert not m.refused
        ref = seed.sift_phase(lattice[i], lattice)
        assert np.array_equal(vm.scores, ref), "VM scores differ from the kernel"
        assert m.topk[:1] == [i]
        assert m.topk == [int(x) for x in np.argsort(-ref, kind="stable")[:7]][: len(m.topk)]
        assert not m.abstain


def test_bind_unbind_exact(world):
    lattice, cb = world
    vm = VM(lattice, cb)
    a, b = lattice[3], lattice[5]
    vm.set(0, a); vm.set(1, b)
    vm.load_program(cb.write([iid("LOAD", 0), iid("BIND", 1), iid("UNBIND", 1)]))
    vm.tick()
    assert np.array_equal(vm.query, a)


def test_tampered_programs_refused(world):
    lattice, cb = world
    vm = VM(lattice, cb)
    good = [iid("LOAD", 0), iid("SIFT", 7), iid("THRESHOLD", 4)]
    vm.load_program(cb.write(good)); vm.set(0, lattice[0]); vm.tick()
    g = np.random.default_rng(0)
    refused = 0
    for _ in range(2000):
        bad = list(good) + [iid("NOP")] * (L - len(good))
        k = int(g.integers(0, L))
        bad[k] = (bad[k] + int(g.integers(1, 120))) % 120
        c = (cb.checksum1(good + [iid("NOP")] * (L - len(good))), cb.checksum2(good + [iid("NOP")] * (L - len(good))))
        vm.load_program(cb.write(bad, check=c))            # old checks, one slot changed
        m = vm.tick()
        refused += int(m.refused)
        assert m.program == vm.marks[0].program            # last good program kept
    assert refused == 2000


def test_prog_rewrite_next_tick(world):
    lattice, cb = world
    vm = VM(lattice, cb)
    p_narrow = cb.write([iid("LOAD", 0), iid("SIFT", 1), iid("THRESHOLD", 7), iid("JUMP", 1)])
    p_wide = cb.write([iid("LOAD", 0), iid("SIFT", 7), iid("THRESHOLD", 2)])
    vm.programs[1] = p_wide
    vm.set(0, lattice[2])
    ok = 0
    for _ in range(100):
        vm.load_program(p_narrow); vm.next_program_vec = None
        m1 = vm.tick(); m2 = vm.tick()
        ok += int("JUMP1" in m1.trace and "SIFT7" in m2.trace and m2.program != m1.program)
    assert ok == 100


def test_rewrite_on_abstain_zoom_lens(world):
    """G4: the sifter changes its own code mid-run. NARROW commits on clear cues; on an ambiguous cue (two identical
    marks -> margin 0) it abstains and PROG rewrites itself to WIDE next tick; WIDE JUMPs back to NARROW. Tampered PROG
    targets are refused and the last good program kept."""
    lattice, cb = world
    amb = np.vstack([lattice, lattice[7:8]])                 # row 20 duplicates row 7 -> ambiguous cue 7
    vm = VM(amb, cb)
    narrow = cb.write([iid("LOAD", 0), iid("SIFT", 1), iid("THRESHOLD", 6), iid("PROG", 1)])
    wide = cb.write([iid("LOAD", 0), iid("SIFT", 7), iid("THRESHOLD", 2), iid("JUMP", 0)])
    vm.programs[0], vm.programs[1] = narrow, wide
    vm.load_program(narrow)
    commits = abst = widened = 0
    g = np.random.default_rng(1)
    for _ in range(200):
        cue = 7 if g.random() < 0.5 else int(g.choice([i for i in range(20) if i != 7]))
        vm.set(0, amb[cue])
        m = vm.tick()
        if cue == 7:
            abst += int(m.abstain)
            if m.abstain:
                m2 = vm.tick()                              # the rewrite took effect: WIDE runs, sees both copies
                widened += int("SIFT7" in m2.trace and set(m2.topk) >= {7, 20})   # WIDE then JUMPs back to NARROW
        else:
            commits += int((not m.abstain) and m.topk == [cue])
    assert abst >= 0.95 * 100 - 5 and widened == abst
    assert commits >= 0.95 * 100 - 5
    # tampered target: a PROG to a program whose check slots do not match is refused, last good program kept
    ids = [iid("LOAD", 0), iid("SIFT", 7), iid("THRESHOLD", 2), iid("JUMP", 0)] + [iid("NOP")] * 4
    bad = list(ids); bad[1] = iid("SIFT", 5)
    vm.programs[1] = cb.write(bad, check=(cb.checksum1(ids), cb.checksum2(ids)))
    vm.load_program(narrow); vm.next_program_vec = None
    vm.set(0, amb[7]); m = vm.tick(); assert m.abstain and "PROG1" in m.trace
    m2 = vm.tick(); assert m2.refused and m2.program == m.program


def test_sift_modes_are_instructions(world):
    """MODE places a sift method on a slot: cascade and soft give the same top-1 as exact on the golden set."""
    lattice, cb = world
    g = np.random.default_rng(5)
    big = g.integers(0, 16, (300, lattice.shape[1])).astype(np.uint8); big[:20] = lattice
    vm = VM(big, cb)
    for mode in (0, 1, 2):
        vm.load_program(cb.write([iid("MODE", mode), iid("LOAD", 0), iid("SIFT", 3), iid("THRESHOLD", 2)]))
        for i in range(16):
            vm.set(0, big[i]); m = vm.tick()
            assert not m.refused and m.topk[:1] == [i], (mode, i, m.topk)


def test_keyed_checks_refuse_public_programs(world):
    """VM-KEYED-0: with a key, a program written with public checksums (a forger without the key) is refused."""
    lattice, _ = world
    kcb = ProgramCodebook(lattice.shape[1], key=b"test-key")
    pub = ProgramCodebook(lattice.shape[1])
    vm = VM(lattice, kcb)
    good = kcb.write([iid("LOAD", 0), iid("SIFT", 3), iid("THRESHOLD", 2)])
    vm.load_program(good); vm.set(0, lattice[1]); assert not vm.tick().refused
    g = np.random.default_rng(0); refused = 0
    for _ in range(2000):
        ids = [int(x) for x in g.integers(0, 120, L)]
        vm.load_program(pub.write(ids)); refused += int(vm.tick().refused)
    assert refused == 2000


def test_tick_memo_exact(world, monkeypatch):
    """IR-JIT-NEXT C1a: the tick memo never changes a score, a top-k or a mark — memo on vs off over a random stream of
    programs, query changes, idle (repeated) ticks, FOCUS regions and appended marks; and idle ticks do hit the memo."""
    lattice, cb = world
    rng = np.random.default_rng(7)
    progs = [cb.write([iid("LOAD", 0), iid("SIFT", 5), iid("THRESHOLD", 3)]),
             cb.write([iid("MODE", 2), iid("LOAD", 0), iid("SIFT", 3)]),
             cb.write([iid("LOAD", 0), iid("BIND", 1), iid("SIFT", 4)]),
             cb.write([iid("FOCUS", 1), iid("LOAD", 0), iid("SIFT", 2)])]
    vms = []
    for flag in ("1", "0"):
        monkeypatch.setenv("OMNIRING_VM_MEMO", flag)
        vms.append(VM(lattice.copy(), cb, regions={1: np.arange(3, 11)}))
    on, off = vms
    assert on.memo and not off.memo
    for step in range(120):
        r = rng.random()
        if r < 0.15:
            p = progs[rng.integers(len(progs))]
            for v in vms: v.load_program(p)
        elif r < 0.30:
            q, q1 = lattice[rng.integers(len(lattice))], lattice[rng.integers(len(lattice))]
            for v in vms: v.set(0, q); v.set(1, q1)
        elif r < 0.36:
            row = seed.codeword(f"new-{step}", 0)
            for v in vms: v.append(row)
        ma, mb = on.tick(), off.tick()                 # anything else = an idle tick (same query again)
        assert ma.topk == mb.topk and ma.abstain == mb.abstain
        assert (on.scores is None) == (off.scores is None)
        if on.scores is not None:
            assert np.array_equal(on.scores, off.scores)
    assert on.memo_hits > 20 and off.memo_hits == 0
    assert np.array_equal(on.lattice, off.lattice)


def test_append_extends_exact_scores(world):
    lattice, cb = world
    vm = VM(lattice.copy(), cb)
    vm.load_program(cb.write([iid("LOAD", 0), iid("SIFT", 4)]))
    vm.set(0, lattice[2]); vm.tick()
    rows = np.stack([seed.codeword(f"x{i}", 0) for i in range(5)])
    vm.append(rows)
    hits = vm.memo_hits; vm.tick()
    assert vm.memo_hits == hits + 1                    # served from the extended memo, no full re-sift
    assert np.array_equal(vm.scores, seed.sift_phase(lattice[2], vm.lattice))
    with pytest.raises(ValueError):
        vm.scores[0] = 0                               # memoised scores are read-only
