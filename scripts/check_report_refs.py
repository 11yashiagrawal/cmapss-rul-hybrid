"""Check the Phase 2 bibliography against the report sections (issue #80).

Reports three problems:

* missing: a key cited in ``reports/phase2/sections/*.tex`` with no entry in
  ``reports/phase2/refs.bib`` (the report would print "[?]");
* duplicate: a key defined twice in ``refs.bib`` (BibTeX errors);
* unused: an entry that no section cites (#80 asks for none).

Exits with status 1 if any problem is found. Run before the final build:

    python scripts/check_report_refs.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "reports" / "phase2"
CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])?\{([^}]*)\}")
ENTRY = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
COMMENT = re.compile(r"(?<!\\)%.*")


def cited_keys(sections: Path) -> dict[str, list[str]]:
    """Every cited key and the section files citing it (LaTeX comments ignored)."""
    cited: dict[str, list[str]] = {}
    for tex in sorted(sections.glob("*.tex")):
        text = COMMENT.sub("", tex.read_text())
        for group in CITE.findall(text):
            for key in filter(None, (k.strip() for k in group.split(","))):
                cited.setdefault(key, []).append(tex.name)
    return cited


def bib_keys(bib: Path) -> list[str]:
    """Entry keys in ``bib``, in file order (repeats kept)."""
    return ENTRY.findall(bib.read_text())


def check(report: Path = REPORT) -> dict[str, list[str]]:
    """``{"missing": [...], "duplicate": [...], "unused": [...]}`` for one report folder."""
    cited = cited_keys(report / "sections")
    defined = bib_keys(report / "refs.bib")
    counts = Counter(defined)
    return {
        "missing": sorted(set(cited) - set(defined)),
        "duplicate": sorted(k for k, n in counts.items() if n > 1),
        "unused": sorted(set(defined) - set(cited)),
    }


def main() -> int:
    problems = check()
    for kind, keys in problems.items():
        print(f"{kind:>9}: {', '.join(keys) if keys else 'none'}")
    return 1 if any(problems.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
