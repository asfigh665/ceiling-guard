# Ceiling-Regret Accounting and Ceiling-Guard

Code and archived experiment outputs for *The Price of Consensus:
Ceiling-Regret Accounting for Collaboration Protocol Design*. The archive
reproduces the five reported evaluation settings from stored responses: MATH-500,
MMLU-Pro, GSM8K, MATH replication, and MMLU replication.

**Authors:** Jiabao Wu, Wenhan Jiang, ZiMeng Li, and Lizheng Zu.

**Repository:** https://github.com/asfigh665/ceiling-guard

## Quick start

Use Python 3.9-3.11. From this directory:

```sh
python -m venv .venv
# Activate the environment using your operating system's standard command.
python -m pip install -r requirements.txt
python analysis/reproduce.py --check
python analysis/validate.py
```

All inputs and reference outputs are included. Evaluation runs offline on CPU;
no model service, GPU, API key, or dataset download is required. Installation of
dependencies requires network access unless they are already available locally.
To regenerate the derived files, run `python analysis/reproduce.py`.

## Contents

- `analysis/reproduce.py`: paired coverage/regret accounting, four readouts,
  guidance comparisons, self-consistency comparisons, bootstrap intervals,
  exact sign-flip tests, Holm correction, and LaTeX table/plot generation.
- `analysis/validate.py`: independent statistical checks, parser edge cases,
  per-item accounting checks, input hashes, and manuscript-reference comparisons.
- `src/`: the answer parsing and normalization functions used by this analysis.
- `data/raw_runs/`: archived debate outputs, sampling outputs, calibration
  accuracies, split IDs, and question sources for replication settings.
- `data/benchmarks/` and `analysis/data/sources/`: benchmark question sources.
- `analysis/data/`: regenerated aggregate and per-item results, provenance hashes,
  and numerical macros. Computational rates use the 0-1 scale.
- `analysis/tables/`: generated LaTeX tables and quantitative plot. Rates and
  absolute changes use the percentage scale; counts and p-values are unchanged.
- `reference/`: frozen numerical results and tables from the manuscript.

## Analysis protocol

Each question's initial and final answers are paired within its stored run.
Replicates are averaged within questions before aggregation and resampling.
Guard uses the multiset of initial and final votes and fixed calibration weights.
Ties are resolved by final-phase support, maximum supporting member weight, and
lexical order. Exact rational arithmetic determines weighted votes.

The bootstrap uses 10,000 question resamples and seed 20261021. Paired p-values
use exact two-sided sign flips. Holm families are recorded in
`analysis/data/manifest_v4.json`. MMLU replication item 432 is excluded because
its stem repeats a main-split question. Ambiguous MMLU option lists abstain;
GSM8K numeric thousands separators are normalized before voting and grading.

Archived calibration accuracies use the original extraction procedure. Raw
calibration responses were not retained and cannot be regraded. This archive
recomputes evaluations from stored responses; it does not rerun model inference.
Original model responses, question texts, numerical results, and research
identifiers are preserved. Local machine paths and development files are not
part of the release.

The illustrated paper figures and manuscript source are delivered separately.
The quantitative plot template here omits the decorative illustration and can
be included in a LaTeX document with TikZ, xcolor, and graphicx loaded. Generated
tables also use booktabs, multirow, and table-color commands as applicable.

## License

The original source code is available under the [MIT License](LICENSE).
Bundled benchmark material, archived model responses, and third-party assets
are excluded from this license grant and remain subject to their original
terms. This repository does not relicense those materials.

Git preserves file bytes (including line endings) so that the SHA-256 release
checksums and provenance hashes remain verifiable after cloning.
