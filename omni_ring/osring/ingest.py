"""Six ingesters onto one Store (OS-RING-0 step 1 of 3, section 7). Each uses only the existing primitives
(residue_address via put_at/get, slot_add/slot_read, add_text/search) in packages/omni-ring/omni_ring/store.py.

IdBook — the one gap found grounding in store.py (CLOSE, not BUILD): Store.slot_add(group, item_id) calls
chain.encode(item_id % DEFAULT_SPEC.data_capacity, rot) — it encodes item_id as a RING COORDINATE, and
data_capacity == 3*5*7*13 == 1,365. That is a slot's legal coordinate space, not a global file-id space: two
items in the *same* slot whose ids collide mod 1,365 are indistinguishable, and BUS-PEEL's read_all_kernel caps
recoverable marks per slot at max_marks (128-160) regardless. So every group here gets its own small LOCAL
counter (0, 1, 2, ...), not a shared global id — IdBook keeps the (group, local_id) -> real key mapping the
engine itself has no slot for, as plain JSON bookkeeping, not a store.py change.

add_text's item_id has no such constraint (store.py only appends it to a python list; it is never encode()'d),
so lattice ingestion (paths' text sample, man pages, journal lines) uses one free-running global counter.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..store import Store

TEXT_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".c", ".h", ".cpp", ".js", ".ts",
             ".rs", ".go", ".java", ".rb", ".cfg", ".ini", ".conf", ".jsonc", ".sql"}


@dataclass
class IdBook:
    """Per-group local indices for slot_add (see module docstring) + a free-running counter for add_text."""
    path: str
    local: Dict[str, Dict[str, int]] = field(default_factory=dict)   # group -> {key: local_id}
    lattice_next: int = 0
    lattice_keys: Dict[int, str] = field(default_factory=dict)      # add_text item_id -> key (for recall checks)

    @classmethod
    def load(cls, path: str) -> "IdBook":
        if os.path.exists(path):
            d = json.load(open(path))
            return cls(path=path, local=d.get("local", {}), lattice_next=d.get("lattice_next", 0),
                       lattice_keys={int(k): v for k, v in d.get("lattice_keys", {}).items()})
        return cls(path=path)

    def save(self) -> None:
        json.dump({"local": self.local, "lattice_next": self.lattice_next, "lattice_keys": self.lattice_keys},
                   open(self.path, "w"))

    def local_id(self, group: str, key: str) -> int:
        """Next local index for `key` within `group` (0-based, per-group — see module docstring)."""
        g = self.local.setdefault(group, {})
        if key not in g:
            g[key] = len(g)
        return g[key]

    def group_size(self, group: str) -> int:
        return len(self.local.get(group, {}))

    def resolve(self, group: str, local_id: int) -> Optional[str]:
        for key, lid in self.local.get(group, {}).items():
            if lid == local_id:
                return key
        return None

    def next_lattice_id(self, key: str) -> int:
        i = self.lattice_next
        self.lattice_next += 1
        self.lattice_keys[i] = key
        return i


def _walk_files(roots: Iterable[str], max_items: int) -> Iterable[Tuple[str, os.stat_result]]:
    n = 0
    for root in roots:
        root = os.path.expanduser(root)
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules", ".venv", "venv")]
            for name in filenames:
                p = os.path.join(dirpath, name)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                yield p, st
                n += 1
                if n >= max_items:
                    return


def ingest_paths(store: Store, book: IdBook, roots: Iterable[str] = ("~", "/usr"), max_items: int = 20_000,
                  max_text_bytes: int = 4096, max_text_items: Optional[int] = None) -> int:
    """address = path (put_at); slot = directory -> entries (slot_add, local index); lattice = name + first
    `max_text_bytes` of text, for files with a recognised text extension (add_text), capped separately at
    `max_text_items` (text_vector's gram encoding is the slow leg — step 2's kernel — so metadata/slot
    ingestion over many more files is decoupled from how many get a lattice entry; default = max_items)."""
    text_cap = max_items if max_text_items is None else max_text_items
    n = 0
    n_text = 0
    for p, st in _walk_files(roots, max_items):
        store.put_at(p, {"size": st.st_size, "mtime": st.st_mtime, "mode": st.st_mode})
        d = os.path.dirname(p)
        store.slot_add(d, book.local_id(d, p))
        base = os.path.basename(p)
        store.slot_add("name:" + base, book.local_id("name:" + base, p))   # extra lookup axis; same primitives
        if n_text < text_cap and os.path.splitext(p)[1].lower() in TEXT_EXTS and st.st_size > 0:
            try:
                with open(p, "r", errors="ignore") as f:
                    sample = base + " " + f.read(max_text_bytes)
            except OSError:
                sample = base
            store.add_text(book.next_lattice_id(p), sample)
            n_text += 1
        n += 1
    return n


def ingest_history(store: Store, book: IdBook, histfile: str = "~/.bash_history", max_items: int = 20_000) -> int:
    """address = command hash (put_at — residue_address hashes the key itself); slot = day (slot_add, local
    index). bash_history here carries no per-line timestamp (no HISTTIMEFORMAT), so every command in one file
    shares the file's mtime-day as its one 'day' bucket — an honest limitation, not a per-command timestamp."""
    histfile = os.path.expanduser(histfile)
    if not os.path.exists(histfile):
        return 0
    import datetime
    day = datetime.date.fromtimestamp(os.path.getmtime(histfile)).isoformat()
    n = 0
    with open(histfile, "r", errors="ignore") as f:
        for line in f:
            cmd = line.rstrip("\n")
            if not cmd or cmd.startswith("#"):
                continue
            store.put_at(cmd, {"day": day})
            store.slot_add(day, book.local_id(day, cmd))
            n += 1
            if n >= max_items:
                break
    return n


def ingest_packages(store: Store, book: IdBook, dpkg_status: str = "/var/lib/dpkg/status", max_items: int = 20_000) -> int:
    """address = package name (put_at); slot = section -> packages (slot_add, local index)."""
    if not os.path.exists(dpkg_status):
        return 0
    n = 0
    name = section = None
    with open(dpkg_status, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("Package: "):
                name = line[len("Package: "):].strip()
            elif line.startswith("Section: "):
                section = line[len("Section: "):].strip()
            elif line.strip() == "" and name:
                if section:
                    store.put_at(name, {"section": section})
                    store.slot_add(section, book.local_id(section, name))
                    n += 1
                    if n >= max_items:
                        return n
                name = section = None
    return n


def ingest_journal(store: Store, book: IdBook, since: str = "7 days ago", max_lines: int = 20_000) -> int:
    """slot = unit -> lines (slot_add, local index); lattice = text (add_text). No address component (section 3
    names none for journal)."""
    try:
        out = subprocess.run(["journalctl", "--user", "-o", "short-iso", "--since", since, "--no-pager"],
                              capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    n = 0
    unit_re = re.compile(r"\S+\[\d+\]:|\S+:")
    for line in out.splitlines():
        parts = line.split(" ", 4)
        unit = parts[4].split(":")[0] if len(parts) >= 5 else "unknown"
        key = f"{line[:24]}|{n}"
        store.slot_add(unit, book.local_id(unit, key))
        store.add_text(book.next_lattice_id(key), line)
        n += 1
        if n >= max_lines:
            break
    return n


def ingest_man(store: Store, book: IdBook, max_pages: int = 5_000) -> int:
    """lattice only (add_text): man page names + their one-line NAME/apropos description."""
    try:
        out = subprocess.run(["man", "-k", "."], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    n = 0
    for line in out.splitlines():
        if not line.strip():
            continue
        name = line.split(" ", 1)[0]
        store.add_text(book.next_lattice_id("man:" + name + f"#{n}"), line)
        n += 1
        if n >= max_pages:
            break
    return n


def ingest_marks(store: Store, book: IdBook, weaver_db: str = "~/.weaver/control-center/field.sqlite",
                  max_rows: int = 20_000) -> int:
    """address = area (put_at, key = mark's `target` field, the closest column to 'area' in the live schema —
    see docs/systems/.../OMNI-INT-S3/S5: lifeos marks write target='lifeos/<area>'); slot = kind -> areas
    (slot_add, local index)."""
    weaver_db = os.path.expanduser(weaver_db)
    if not os.path.exists(weaver_db):
        return 0
    con = sqlite3.connect(weaver_db)
    try:
        rows = con.execute("SELECT kind, target FROM marks WHERE target != '' ORDER BY ts DESC LIMIT ?",
                            (max_rows,)).fetchall()
    finally:
        con.close()
    n = 0
    for kind, target in rows:
        area = target
        store.put_at(area, {"kind": kind})
        store.slot_add(kind, book.local_id(kind, area))
        n += 1
    return n


def ingest_all(store: Store, book: IdBook, roots: Iterable[str] = ("~", "/usr"), path_cap: int = 20_000,
               text_cap: Optional[int] = None, history_cap: int = 20_000, package_cap: int = 20_000,
               journal_cap: int = 20_000, man_cap: int = 5_000, marks_cap: int = 20_000) -> Dict[str, int]:
    counts = {
        "paths": ingest_paths(store, book, roots, path_cap, max_text_items=text_cap),
        "history": ingest_history(store, book, max_items=history_cap),
        "packages": ingest_packages(store, book, max_items=package_cap),
        "journal": ingest_journal(store, book, max_lines=journal_cap),
        "man": ingest_man(store, book, max_pages=man_cap),
        "marks": ingest_marks(store, book, max_rows=marks_cap),
    }
    store.commit()
    book.save()
    return counts
