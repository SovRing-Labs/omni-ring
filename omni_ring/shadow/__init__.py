"""SOV-SHADOW-0 — Sov Raptor's field state on the omniring Store, SHADOW ONLY (2026-10-02).
Own package: imports nothing from launcher_core; reads a COPY of ~/.weaver/control-center/field.sqlite (read-only);
an absent store or db => empty results, never an exception (0.1.0 ships standalone)."""
