# Phase 2 report sections

[dl_dataset_preview.pdf](dl_dataset_preview.pdf) is the standalone IEEE-format
preview for issue #74. It contains the sequence-dataset/regularization section,
two tables, the #66 learning-curve figure, and its six references. It is a section
preview, not the assembled Phase 2 report.

## Build and integration

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

## Evidence and reproduction

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
