"""The user guide passes ``tools/check_guide.py``, and the checker can fail.

``plans/user-guide-web.md`` section 6: every ``Calculator`` method and every
README feature row is on a page, every include and image resolves, every page
is in the nav, and no prose carries a semicolon or an em-dash. The second test
is the guard on the guard: a checker that read nothing would also pass, so it is
run on a small tree built to fail each check and must name each failure.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "check_guide.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_guide", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_guide_passes_check_guide():
    run = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_check_guide_names_each_kind_of_failure(tmp_path):
    (tmp_path / "defuqumat").mkdir()
    methods = "\n".join(f"    def get_m{i}(self): pass" for i in range(40))
    (tmp_path / "defuqumat" / "calculator.py").write_text(
        f"class Calculator:\n{methods}\n    def get_missing(self): pass\n")
    rows = "\n".join(f"| **Row {i}** | `calc.get_m{i}()` | | |" for i in range(40))
    (tmp_path / "README.md").write_text(
        "| Quantity | How | QE | Elk |\n|---|---|:-:|:-:|\n" + rows
        + "\n| **Robust** | `get_m0(mixing_mode='robust')` | | |\n")
    guide = tmp_path / "docs" / "guide"
    (guide / "img").mkdir(parents=True)
    (guide / "img" / "a.png").write_bytes(b"")
    calls = "\n".join(f"calc.get_m{i}()" for i in range(40))
    (guide / "index.md").write_text(
        f"# Home\n\n```python\n{calls}\ncalc.get_gone()\n```\n\n"
        "Code `a;b` and math $a;b$ are fine; this is not.\n\n"
        "Nor is this — here.\n\n"
        "--8<-- \"nowhere/compute.py\"\n\n"
        "![x](img/a.png#only-light)\n")
    (guide / "orphan.md").write_text("# Orphan\n")
    (tmp_path / "mkdocs.yml").write_text(
        "markdown_extensions:\n  - pymdownx.snippets:\n"
        "      base_path: [docs/figures]\n"
        "nav:\n  - Home: index.md\n  - Gone: gone.md\n")

    failures = _load().check(tmp_path)
    text = "\n".join(m for ms in failures.values() for m in ms)
    assert "get_missing is on no page" in text
    assert "names get_gone" in text
    assert "Row 0" not in text
    assert "README row 'Robust'" in text
    assert "nowhere/compute.py does not resolve" in text
    assert "no dark partner img/a-dark.png" in text
    assert "orphan.md is not in the nav" in text
    assert "nav entry gone.md is not a page" in text
    assert "a semicolon in prose: Code  and math  are fine; this is" in text
    assert "an em-dash in prose: Nor is this" in text
    assert len(failures["prose"]) == 2
