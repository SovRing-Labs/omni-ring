"""OMNIRING-TWIN-1 drift gate: ~/projects/omniring-llm/src/<mod>.py must stay an alias of omni_ring.<mod>,
and the project's tests must not diverge from these (modulo the src. -> omni_ring. import prefix)."""
from pathlib import Path
import importlib

import pytest

PROJ = Path.home() / "projects" / "omniring-llm"
HERE = Path(__file__).resolve().parent


@pytest.mark.skipif(not PROJ.exists(), reason="project tree not present")
def test_project_src_modules_are_aliases():
    for f in sorted((PROJ / "src").glob("*.py")):
        if f.stem == "__init__":
            continue
        assert "_sys.modules[__name__] = _mod" in f.read_text(), f"{f} is a copy, not an alias of omni_ring.{f.stem}"
        importlib.import_module(f"omni_ring.{f.stem}")


@pytest.mark.skipif(not PROJ.exists(), reason="project tree not present")
def test_project_tests_match_package_tests():
    for f in sorted((PROJ / "tests").glob("test_*.py")):
        mine = HERE / f.name
        assert mine.exists(), f"{f.name} exists only in the project tree"
        norm = lambda s: s.replace("from src.", "from omni_ring.").replace("import src.", "import omni_ring.")  # noqa: E731
        a, b = norm(f.read_text()), norm(mine.read_text())
        if a != b:
            # test_mark_store keeps a sys.path shim in the project copy; compare test bodies only
            strip = lambda s: "\n".join(l for l in s.splitlines() if l.startswith("def test") or l.startswith("    assert"))  # noqa: E731
            assert strip(a) == strip(b), f"{f.name} drifted between project and package"
