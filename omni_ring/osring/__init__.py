"""OS-RING-0 (2026-10-02) — a legacy machine's whole state on the ring. Six ingesters write a laptop's file paths,
shell history, dpkg status, journal, man pages and Weaver field marks into one omni_ring.store.Store, through the
existing address/slot/lattice primitives only (CLOSE on store.py — see ingest.py's IdBook for the one thing it is
missing: a per-group *local* index, because Store.slot_add encodes item_id via chain.encode(item_id % data_capacity,
...), and data_capacity is 1,365 — a slot's legal coordinate space, not a global file-id space)."""
from .ingest import IdBook, ingest_all, ingest_history, ingest_journal, ingest_man, ingest_marks, ingest_packages, ingest_paths

__all__ = ["IdBook", "ingest_all", "ingest_history", "ingest_journal", "ingest_man", "ingest_marks", "ingest_packages", "ingest_paths"]
