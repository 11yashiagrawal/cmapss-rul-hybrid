# Phase 2 report

`main.tex` is the IEEE conference report for Phase 2 (#81). Like Phase 1, it sets up the
IEEEtran template and inputs one file per section from `sections/`, in report order. Each
section file has one owner, assigned through its GitHub issue.

| Order | File in `sections/` | Issue | Owner |
|---|---|---|---|
| 1 | `abstract.tex`, `introduction.tex` | #81 | open |
| 2 | `literature_review.tex` (+ `.bib`) | #63 | 11yashiagrawal |
| 3 | `target.tex` (subsection under *Training Target and Model Selection*) | #55 | 11yashiagrawal |
| 4 | `dl_dataset.tex` (+ `.bib`) | #74 | affan80 |
| 5 | `theory.tex` (+ `.bib`) | #67 | 11yashiagrawal |
| 6 | `architecture.tex`, `hybrid.tex` | #81 | open |
| 7 | `ablations.tex` | #75 | Bytebard089 |
| 8 | `validation.tex`, `interpretability.tex` | #76 | MilindDevX |
| 9 | `results.tex`, `conclusion.tex`, `contributions.tex` | #81 | open |

Write your username next to a #81 part in the issue before you start it (#83 asks for every
part to have an owner by Sat 14 Nov), and update the table here in your PR.

## Building while sections are still landing

`main.tex` inputs a section only if its file exists. A missing file prints a bold
**[Placeholder: sections/<file>.tex, issue #n]** in its place, so you can build the report
with just your own section. Section files use `\section{...}`, except `target.tex`, which is
a subsection under a heading in `main.tex`.

References: until #80 adds `refs.bib`, `main.tex` uses `../phase1/refs.bib` plus every
`sections/*.bib` fragment that exists. Until then BibTeX reports a few *repeated entries*
(the same key in two fragments); #80 removes them. Once `refs.bib` exists it is used alone.

Figures are read from `../../results/figures/`, so build inside a full checkout.

From this folder, with [Tectonic](https://tectonic-typesetting.github.io/):

```sh
tectonic main.tex
```

Or with TeX Live: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Before the final build (#81)

- [ ] No `Placeholder` left: `grep -c Placeholder main.log` prints 0.
- [ ] No undefined references or citations, and no BibTeX warnings (after #80).
- [ ] No overfull boxes: `tectonic main.tex 2>&1 | grep -i overfull` prints nothing.
- [ ] Every number checked against `results/metrics/`.
- [ ] `main.pdf` is under 10 MB; commit it with the final sources.

## Section preview: DL dataset and regularization (#74)

[dl_dataset_preview.pdf](dl_dataset_preview.pdf) is the standalone IEEE-format
preview for issue #74. It contains the sequence-dataset/regularization section,
two tables, the #66 learning-curve figure, and its six references. It is a section
preview, not the assembled Phase 2 report.

### Build and integration

From this directory:

```sh
tectonic --keep-logs dl_dataset_preview.tex
```

For TeX Live, run `pdflatex dl_dataset_preview`, `bibtex dl_dataset_preview`,
then `pdflatex dl_dataset_preview` twice. Build in a complete repository checkout;
the figure is read from `../../results/figures/`. No raw dataset is needed to
build the PDF because the verified tables and figure are committed.

For the final report, input `sections/dl_dataset.tex` from `reports/phase2/`,
with `amsmath`, `graphicx`, `cite`, and `url` available in the preamble. Issue #80
should consolidate `sections/dl_dataset.bib` into `refs.bib`; all six keys are
cited, and the two existing Phase 1 keys are reused. The fragment and preview
avoid taking over the bibliography-consolidation task. Rebuild the preview
whenever its section, tables, figure, or bibliography changes.

### Evidence and reproduction

The report's study CSV and figure are byte-identical snapshots from issue #66,
PR #97, commit `598ecb8dce4d101d118b23dd6ef6d2a1790f2615`. They are included here
so the section builds while #97 awaits integration. This PR carries only those
small shared assets, not the study implementation or large prediction archive.
Merging #97 first remains the preferred integration order.

With the project environment installed and FD001 training data downloaded, run
from the repository root:

```sh
python scripts/prepare_dl_dataset_report.py          # regenerate tables/data
python scripts/prepare_dl_dataset_report.py --check  # verify without writing
```

The script opens only `train_FD001.txt`, uses the project's real engine splits
and window builder, checks the closed-form counts, and records engine lists,
sensor selection, raw-data hash, package versions, and #66 asset hashes in
`data/dl_dataset_provenance.json`. It stops if the study assets differ from the
recorded snapshot. `data/dl_window_counts.csv` holds the numeric window counts;
`tables/` contains the generated LaTeX tabular bodies. Edit source/configuration
and regenerate rather than hand-editing numeric tables.

The results remain **provisional until #55** finalizes the cap and selection
rule: current training cap 125 and true-RUL selection threshold 130 match #58
and #66. A changed threshold can be applied to #66's saved predictions; a
changed training cap requires retraining. Refresh the snapshot, recorded hashes,
tables and prose when those results change. The report does not claim that a
single-seed grid establishes statistical significance or final frozen settings.
