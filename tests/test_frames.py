"""Frames: one vector unfolds into its parts (Plate's 'Mark ate the fish', nested in 'Hunger caused ...')."""
from omni_ring.frames import Frame, ItemMemory


def test_plate_frames_unfold():
    im = ItemMemory(d=1024)
    s1 = Frame("eat", agt="mark", obj="the_fish")
    v1 = im.encode(s1, chunk_as="s1")
    v2 = im.encode(Frame("cause", agt="hunger", obj=s1), chunk_as="s2")
    for a in ("john", "paul", "luke", "fish", "food", "thirst"):
        im.add_atom(a)
    assert im.cleanup(im.unbind(v1, "agt"))[0][0] == "mark"
    assert im.cleanup(im.unbind(v2, "obj"))[0][0] == "s1"
    assert im.unfold(v2, ["obj", "agt"])[0][0] == "mark"
    assert im.unfold(v2, ["obj", "obj"], clean_between=False)[0][0] == "the_fish"
