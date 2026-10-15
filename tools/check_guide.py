#!/usr/bin/env python3
"""Check the user guide, ``docs/guide/*.md``, against the code and against itself.

``plans/user-guide-web.md`` section 6: coverage is a set difference, not a
read-through. Four checks, each printed under its own heading:

1. **Coverage.** Every ``get_*``, ``with_*`` and ``from_*`` method of
   ``Calculator`` and every row of the README feature table appears on at
   least one page, as code: inside a fenced block, an inline code span, or a
   file a page pulls in with ``--8<--`` (that file is the snippet the reader
   sees). A README row is covered when, for at least one backticked item of its
   "How to ask for it" column, every identifier of that item is in the code of
   one page, so ``get_scf(mixing_mode='robust')`` needs ``get_scf``,
   ``mixing_mode`` and ``robust``. The other direction too: a page calling
   ``calc.get_x``, ``calc.with_x`` or ``Calculator.from_x`` that does not exist.
2. **Files.** Every ``--8<--`` include resolves against the snippets
   ``base_path`` of ``mkdocs.yml``, every image exists, and every
   ``#only-light`` image has its ``-dark`` partner on the same page (and the
   other way round), both on disk.
3. **Nav.** Every page is in the ``nav`` of ``mkdocs.yml`` and every nav entry
   is a page.
4. **Prose.** No semicolon and no em-dash outside code, math, link targets,
   HTML entities and include lines. And no ``[defuqumat] ...`` line in a page
   at all: that is the announcement the code prints on stderr when it runs the
   SCF first, and a printed output block on a page holds stdout only.

It reads files only: ``calculator.py`` is parsed with ``ast``, nothing from
``defuqumat`` is imported, so it runs in well under a second.

    python3 tools/check_guide.py          # exit status 1 if anything fails
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("get_", "with_", "from_")

FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
INCLUDE = re.compile(r"^[ \t]*--8<--[ \t]+[\"']?([^\"'\s]+)[\"']?")
INLINE_CODE = re.compile(r"(`+)(.+?)\1")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
IDENT = re.compile(r"[A-Za-z_][\w-]*")
CALL = re.compile(r"\b(?:calc|Calculator)\.((?:get|with|from)_\w+)")
NOT_A_KEY = {"calc", "Calculator"}


def calculator_methods(root: Path) -> set[str]:
    tree = ast.parse((root / "defuqumat" / "calculator.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Calculator":
            return {f.name for f in node.body
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and f.name.startswith(PREFIXES)}
    return set()


def item_keys(item: str) -> set[str]:
    """The identifiers a README entry-point item needs to see on a page.

    Attribute and method names, keyword names, string values and bare names,
    but not a positional argument of a call, which is a placeholder
    (``get_effective_mass(kpoint)`` needs ``get_effective_mass`` only).
    An item that is not Python (``tefield, dipfield`` or ``grimme-d2`` outside
    quotes) falls back to every identifier in it.
    """
    try:
        tree = ast.parse(item.replace("...", "None"))
    except SyntaxError:
        return {k for k in IDENT.findall(item) if k not in NOT_A_KEY}
    placeholders = {id(a) for node in ast.walk(tree) if isinstance(node, ast.Call)
                    for a in node.args if isinstance(a, ast.Name)}
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            keys.add(node.attr)
        elif isinstance(node, ast.Name) and id(node) not in placeholders:
            keys.add(node.id)
        elif isinstance(node, ast.keyword) and node.arg:
            keys.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            keys.add(node.value)
    return keys - NOT_A_KEY


def readme_rows(root: Path) -> list[tuple[str, list[str]]]:
    """``(quantity, [backticked items of the entry-point column])`` per row."""
    rows = []
    for line in (root / "README.md").read_text().splitlines():
        if not line.startswith("| **"):
            continue
        cells = re.split(r"(?<!\\)\|", line)
        if len(cells) < 4:
            continue
        quantity = re.sub(r"\*\*(.+?)\*\*.*", r"\1", cells[1].strip())
        items = [m.group(2) for m in INLINE_CODE.finditer(cells[2])]
        rows.append((quantity, items))
    return rows


def split_page(text: str):
    """Return ``(code, prose_lines)``: the page's code, and its prose by line.

    Fenced blocks may be indented (inside an admonition), so the fence is
    matched after leading whitespace; an indented line is not code here,
    since it is how an admonition body is written.
    """
    code, prose = [], []
    fence = None
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if fence is not None:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            else:
                code.append(line)
            prose.append((n, ""))
            continue
        if m:
            fence = m.group(1)
            prose.append((n, ""))
            continue
        code.extend(m.group(2) for m in INLINE_CODE.finditer(line))
        prose.append((n, line))
    return "\n".join(code), prose


def strip_prose(prose):
    """Remove math, inline code, link targets, entities and include lines."""
    text = "\n".join(line for _, line in prose)
    text = re.sub(r"\$\$.*?\$\$", lambda m: "\n" * m.group(0).count("\n"),
                  text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]",
                  lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    out = []
    for (n, _), line in zip(prose, text.split("\n")):
        if INCLUDE.match(line):
            continue
        line = INLINE_CODE.sub("", line)
        line = re.sub(r"(?<!\\)\$[^$]+?(?<!\\)\$", "", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        line = re.sub(r"&#?\w+;", "", line)
        line = re.sub(r"<[^>]+>", "", line)
        out.append((n, line))
    return out


def nav_pages(nav) -> list[str]:
    found = []
    if isinstance(nav, str):
        found.append(nav)
    elif isinstance(nav, list):
        for item in nav:
            found += nav_pages(item)
    elif isinstance(nav, dict):
        for value in nav.values():
            found += nav_pages(value)
    return found


def check(root: Path = ROOT) -> dict[str, list[str]]:
    """Every failure, keyed by check; an empty dict means the guide passes."""
    guide = root / "docs" / "guide"
    config = yaml.safe_load((root / "mkdocs.yml").read_text())
    base = ["docs/figures"]
    for ext in config.get("markdown_extensions", []):
        if isinstance(ext, dict) and "pymdownx.snippets" in ext:
            base = ext["pymdownx.snippets"].get("base_path", base)
    base = [root / b for b in ([base] if isinstance(base, str) else base)]

    failures = {"coverage": [], "files": [], "nav": [], "prose": []}
    pages = sorted(guide.glob("*.md"))
    page_code = {}

    for page in pages:
        text = page.read_text()
        code, prose = split_page(text)
        name = page.name
        for n, line in enumerate(text.splitlines(), 1):
            m = INCLUDE.match(line)
            if m:
                hits = [b / m.group(1) for b in base if (b / m.group(1)).is_file()]
                if hits:
                    code += "\n" + hits[0].read_text()
                else:
                    failures["files"].append(
                        f"{name}:{n}: include {m.group(1)} does not resolve")
        page_code[name] = code

        images = {}
        for n, line in enumerate(text.splitlines(), 1):
            for m in IMAGE.finditer(line):
                target, _, frag = m.group(1).partition("#")
                if "://" in target:
                    continue
                images.setdefault(frag, []).append((n, target))
                if not (guide / target).is_file():
                    failures["files"].append(
                        f"{name}:{n}: image {target} does not exist")
        light = {t for _, t in images.get("only-light", [])}
        dark = {t for _, t in images.get("only-dark", [])}
        for n, t in images.get("only-light", []):
            partner = re.sub(r"(\.\w+)$", r"-dark\1", t)
            if partner not in dark:
                failures["files"].append(
                    f"{name}:{n}: {t} has no dark partner {partner}")
        for n, t in images.get("only-dark", []):
            partner = re.sub(r"-dark(\.\w+)$", r"\1", t)
            if partner not in light:
                failures["files"].append(
                    f"{name}:{n}: {t} has no light partner {partner}")

        for n, line in enumerate(text.splitlines(), 1):
            if line.startswith("[defuqumat] "):
                failures["prose"].append(
                    f"{name}:{n}: a stderr announcement in an output block: {line[:60]}")

        for n, line in strip_prose(prose):
            for ch, what in ((";", "a semicolon"), ("—", "an em-dash")):
                if ch in line:
                    failures["prose"].append(
                        f"{name}:{n}: {what} in prose: {line.strip()[:80]}")

    methods = calculator_methods(root)
    if len(methods) < 30:
        failures["coverage"].append(
            f"only {len(methods)} Calculator methods parsed; has calculator.py moved?")
    everywhere = "\n".join(page_code.values())
    for method in sorted(methods):
        if not re.search(rf"\b{method}\b", everywhere):
            failures["coverage"].append(f"Calculator.{method} is on no page")
    for name, code in page_code.items():
        for called in sorted(set(CALL.findall(code)) - methods):
            failures["coverage"].append(
                f"{name}: names {called}, which Calculator does not have")

    rows = readme_rows(root)
    if len(rows) < 40:
        failures["coverage"].append(
            f"only {len(rows)} README feature rows parsed; has the table moved?")
    for quantity, items in rows:
        keys = [k for k in map(item_keys, items) if k]
        covered = any(
            all(re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", code) for k in ks)
            for ks in keys for code in page_code.values())
        if not covered:
            failures["coverage"].append(
                f"README row '{quantity}' ({', '.join(items)}) is on no page")

    listed = nav_pages(config.get("nav", []))
    on_disk = {p.name for p in pages}
    for page in sorted(on_disk - set(listed)):
        failures["nav"].append(f"{page} is not in the nav of mkdocs.yml")
    for page in sorted(set(listed) - on_disk):
        failures["nav"].append(f"nav entry {page} is not a page")

    return {k: v for k, v in failures.items() if v}


def main() -> int:
    failures = check(ROOT)
    if not failures:
        print("check_guide: coverage, files, nav and prose all pass")
        return 0
    for kind, messages in failures.items():
        print(f"{kind}: {len(messages)}")
        for message in messages:
            print(f"  {message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
