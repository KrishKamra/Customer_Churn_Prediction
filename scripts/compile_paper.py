"""Compile docs/paper/main.tex to docs/paper/main.pdf with a local TeX engine.

Builds in a temporary directory (pdfLaTeX -> BibTeX -> pdfLaTeX x2, plus extra passes while
LaTeX asks for a rerun), fails on any LaTeX error or unresolved reference, reports overfull
boxes, and copies the PDF back. Run ``scripts/build_paper.py`` first to refresh the
generated tables and charts.

Run: uv run python scripts/compile_paper.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "docs" / "paper"
MIKTEX_USER_BIN = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64"


def _tool(name: str) -> str:
    found = shutil.which(name) or shutil.which(name, path=str(MIKTEX_USER_BIN))
    if not found:
        sys.exit(f"'{name}' not found. Install a TeX distribution (e.g. MiKTeX or TeX Live).")
    return found


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", check=False)


def main() -> int:
    pdflatex, bibtex = _tool("pdflatex"), _tool("bibtex")
    latex = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    with tempfile.TemporaryDirectory(prefix="churnai_paper_") as tmp:
        build = Path(tmp)
        for item in ("main.tex", "references.bib", "generated", "figures"):
            src = PAPER / item
            (shutil.copytree if src.is_dir() else shutil.copy)(src, build / item)

        _run(latex, build)
        _run([bibtex, "main"], build)
        for _ in range(4):  # extra passes until cross-references settle
            _run(latex, build)
            log = (build / "main.log").read_text(encoding="latin-1")
            if not re.search(r"Rerun to get|may have changed", log):
                break

        errors = re.findall(r"(?m)^! .*$", log)
        unresolved = re.findall(r"(?m)^.*(?:Citation|Reference) `[^']+' .*undefined.*$", log)
        if errors or unresolved or not (build / "main.pdf").exists():
            print("\n".join(errors + unresolved) or "no PDF produced", file=sys.stderr)
            return 1
        for box in re.findall(r"(?m)^Overfull \\[hv]box.*$", log):
            print("warning:", box)
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
        shutil.copy(build / "main.pdf", PAPER / "main.pdf")
    size = (PAPER / "main.pdf").stat().st_size / 1024
    print(f"wrote {PAPER / 'main.pdf'} ({pages.group(1) if pages else '?'} pages, {size:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
