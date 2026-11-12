"""Tests for scripts/check_report_refs.py and the Phase 2 bibliography (#80)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_report_refs.py"


@pytest.fixture(scope="module")
def refs():
    spec = importlib.util.spec_from_file_location("check_report_refs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_report(tmp_path, sections: dict, bib: str) -> Path:
    (tmp_path / "sections").mkdir()
    for name, text in sections.items():
        (tmp_path / "sections" / name).write_text(text)
    (tmp_path / "refs.bib").write_text(bib)
    return tmp_path


def test_finds_missing_duplicate_and_unused_keys(refs, tmp_path):
    report = make_report(
        tmp_path,
        {"a.tex": r"See~\cite{Known, Missing} and \citep[p.~2]{Twice}.",
         "b.tex": r"% \cite{OnlyInComment}" "\n" r"Also \cite{Known}."},
        "@article{Known,\n}\n@misc{Twice,\n}\n@misc{Twice,\n}\n@book{Unused,\n}\n",
    )

    assert refs.check(report) == {
        "missing": ["Missing"], "duplicate": ["Twice"], "unused": ["Unused"],
    }


def test_escaped_percent_is_not_a_comment(refs, tmp_path):
    report = make_report(tmp_path, {"a.tex": r"50\% of engines~\cite{Key}"}, "@misc{Key,\n}\n")

    assert refs.check(report) == {"missing": [], "duplicate": [], "unused": []}


def test_phase2_bibliography_covers_every_citation_once(refs):
    problems = refs.check()

    # Missing keys would print "[?]" and duplicates break BibTeX. Unused entries
    # can appear while their sections are still in open PRs, so they are only
    # enforced by the script before the final build (#81).
    assert problems["missing"] == []
    assert problems["duplicate"] == []
