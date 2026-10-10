"""Store round-trip, including the "file" slot-group name that collided with np.savez's own `file` kwarg
(OS-RING-0 ingest_marks hits this for real: Weaver mark kind == "file")."""
import numpy as np

from omni_ring.store import Store


def test_address_roundtrip(tmp_path):
    st = Store(str(tmp_path / "s1"))
    st.put_at("/a/b.txt", {"size": 3})
    assert st.get("/a/b.txt") == {"key": "/a/b.txt", "size": 3}
    assert st.get("/missing") is None


def test_slot_group_named_file_survives_commit_and_reload(tmp_path):
    p = str(tmp_path / "s2")
    st = Store(p)
    st.slot_add("file", 0)
    st.slot_add("file", 1)
    st.slot_add("other", 2)
    st.commit()
    st2 = Store(p)
    assert set(st2._slots.keys()) == {"file", "other"}
    assert np.any(st2._slots["file"])


def test_lattice_search_roundtrip(tmp_path):
    st = Store(str(tmp_path / "s3"))
    st.add_text(0, "alpha beta gamma")
    st.add_text(1, "completely unrelated words here")
    st.commit()
    st2 = Store(str(tmp_path / "s3"))
    hits = st2.search("alpha beta", k=1)
    assert hits and hits[0][0] == 0
