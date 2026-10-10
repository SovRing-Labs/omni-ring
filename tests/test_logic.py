"""LOGIC-0 spine + Lexicon: exact chaining, proofs, contradictions, DRed retraction == recomputation, calibrated abstain."""
from omni_ring.logic import Lexicon, Tracker

RULES = [(("ancestor", "X", "Y"), [("parent", "X", "Y")]),
         (("ancestor", "X", "Z"), [("parent", "X", "Y"), ("ancestor", "Y", "Z")]),
         (("elder", "X"), [("ancestor", "X", "Y"), ("child", "Y")])]
FACTS = [("parent", "a", "b"), ("parent", "b", "c"), ("parent", "c", "d"), ("child", "d")]


def build(facts=FACTS):
    t = Tracker(RULES)
    for f in facts:
        t.base.add(f); t._add(f)
    t.chain()
    return t


def test_chain_depth_and_proof():
    t = build()
    assert ("ancestor", "a", "d") in t.facts and ("elder", "a") in t.facts
    assert t.depth(("ancestor", "a", "d")) == 3
    p = t.proof(("elder", "a"))
    assert p["rule"] == 2 and len(p["premises"]) == 2


def test_contradiction_both_sides():
    t = build()
    t.base.add(("not", "elder", "a")); t._add(("not", "elder", "a"))
    t.base.add(("not", "elder", "zz")); t._add(("not", "elder", "zz"))
    c = t.contradictions()
    assert [x[0] for x in c] == [("elder", "a")]


def test_retraction_equals_recomputation():
    for f in FACTS:
        t = build(); t.retract(f)
        assert t.facts == build([g for g in FACTS if g != f]).facts


def test_lexicon_calibrated_abstain():
    L = Lexicon()
    for s, c in [("parent", "P"), ("is_parent", "P"), ("child", "C"), ("is_child", "C")]:
        L.add(s, c)
    L.calibrate([("parents", "P"), ("childs", "C")], ["xylophone", "quartz"])
    assert L.resolve("parent_of")[0] == "P" and L.resolve("zebra")[0] is None


def test_prover_backward_matches_forward_and_rechecks():
    from omni_ring.logic import Prover, check_proof
    fwd = build()
    t = Tracker(RULES)
    for f in FACTS:
        t.base.add(f); t._add(f)                                  # base only: the prover must derive the middle itself
    pr = Prover(t)
    for goal in [("elder", "a"), ("ancestor", "a", "d"), ("elder", "d"), ("ancestor", "d", "a")]:
        tree = pr.prove(goal)
        assert (tree is not None) == (goal in fwd.facts)
        if tree is not None:
            assert check_proof(t, tree)
