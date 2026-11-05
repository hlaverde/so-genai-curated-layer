# The Curated Layer Shrinks Too: Generative AI and Public Knowledge Production on Stack Overflow

Replication package for the article by **Henry Laverde-Rojas** and
**Carlos Alfonso Laverde-Rodríguez** (Programa de Economía, Universidad
Militar Nueva Granada, Bogotá, Colombia). Release v2.0,
<https://github.com/hlaverde/so-genai-curated-layer>.

## Overview

The code builds a tag × week × question-type panel of Stack Overflow questions from
public sources, on the full grid of tags, weeks and question types with zero counts in
cells that have no question, and produces every estimate, table and figure of the
article and its Online Resource 1. There are two ways to replicate them:

1. **Quick replication from `data/analysis/`.** The analysis data are part of this
   package, so `python run_pipeline.py --from-analysis-data` downloads nothing. It takes
   50 to 95 minutes on the reference machine.
2. **Full replication from the sources.** The download scripts and SEDE queries rebuild
   `data/analysis/` from the public sources, and `python run_pipeline.py` then runs the
   analysis. A new download can differ from the data of the article (see "Collection").

Both write `results/numbers.tex` (one LaTeX macro per number quoted in the text), the
tables in `results/tables/`, the figures in `results/figures/` and the underlying
estimates in `results/estimates/`.

## Data availability and provenance statements

### Statement about rights

- [x] The authors have legitimate access to and permission to use the
  data used in the article.
- [x] The authors have permission to redistribute the data contained in
  this package. Only derived data are redistributed: cell-level panels, tag-level
  inputs, and question-level variables identified by question ID. No title, body or
  other question text and no user identifier is included.

### License for data

Stack Overflow content is published by Stack Exchange, Inc. under the Creative Commons
Attribution-ShareAlike 4.0 license (CC BY-SA 4.0). The derived data in `data/analysis/`
and `results/` are distributed under the same license, with attribution to Stack
Exchange, Inc. and the contributors to Stack Overflow. The code is distributed under the
MIT license. `LICENSE` states both.

### Summary of availability

- [x] All data are publicly available.
- [ ] Some data cannot be made publicly available.
- [ ] No data can be made publicly available.

The analysis data are in this package. The raw inputs are public and are not included:
the scripts in `sql/` and `src/data/` regenerate them from the sources below.

### Details on each data source

| Data source | Access | Files produced | Included |
|---|---|---|---|
| Stack Exchange Data Explorer (SEDE), Stack Overflow database, queried through the browser at <https://data.stackexchange.com/stackoverflow> | Free; queries pasted manually (SEDE requires an interactive session) | Question–tag rows 2020-01 to 2023-05; tag-week aggregates; tag-week complexity aggregates; question-level validation pool | Derived data only |
| Stack Exchange API v2.3, <https://api.stackexchange.com> | Free; a registered application key (no payment) raises the request quota | Question–tag rows 2023-06 to 2024-12; all of 2025; the 2023-05-02 day used to validate SEDE against the API | Derived data only |
| Sentence-embedding models `sentence-transformers/all-MiniLM-L6-v2` and `all-mpnet-base-v2` (Hugging Face) | Free; downloaded automatically on first use by the full replication | Embedding classifiers and the embedding-based answerability score | Labels and scores only |

Data citation: Stack Exchange, Inc. Stack Exchange Data Explorer and
Stack Exchange API v2.3, Stack Overflow site, accessed May and June 2026.

### Collection

The data of the article were collected from 15 May to 2 June 2026: the SEDE extracts
from 15 to 21 May, the API rows of 2023-06 to 2024-12 and of the validation day on 21 and
22 May, and the API rows of 2025 on 2 June. Stack Overflow changes over time: questions
are deleted, votes accumulate, answers are accepted and questions are closed or
reopened, and SEDE refreshes its copy of the database periodically. A new download
therefore can differ from these data, and its estimates can differ from those of the
article. `data/analysis/` holds the data of the article.

### The `selenium` tag

The raw extracts contain no question of the tag `selenium` created from 2023-02-13 to
2023-02-28, so that tag's cells are zero or partial in the three weeks this period
touches. Its questions of March 2023 come from a separate SEDE backfill on its alias tags
(`selenium-webdriver`, `selenium-chromedriver`, `webdriver`, `chromedriver`), counted
under `selenium` (`sql/sede_question_type_raw_windows/*_selenium_alias_backfill.sql`),
and the 2025 collection pulls the tag through the same aliases. The step
`estimates.selenium_gap` writes to `results/numbers.tex` the estimates without those
weeks, without the backfilled pairs and without the tag (macros `SelDrop*`, `SelNoBack*`,
`SelNoTag*`), and the article reports them.

## Dataset list

### In this package

`data/analysis/README.md` defines every variable.

| File | Level | Rows |
|---|---|---|
| `data/analysis/panel_main.csv.gz` | Tag × week × question type, main window | 182,700 |
| `data/analysis/panel_extended.csv.gz` | Tag × week × question type, with 2025 | 218,400 |
| `data/analysis/question_tag.csv.gz` | Question × tag | 8,511,130 |
| `data/analysis/minilm_panel_title_labels.csv.gz` | Question | 4,987,056 |
| `data/analysis/validation_sample.csv.gz` | Question | 9,996 |
| `data/analysis/revealed_pool.csv.gz` | Question × tag | 50,000 |
| `data/analysis/selenium_backfill.csv.gz` | Question | 793 |
| `data/analysis/tag_index.csv.gz` | Tag | 100 |
| `data/analysis/tag_embedding_score.csv.gz` | Tag | 100 |
| `data/analysis/question_type_substitutability.csv.gz` | Question type | 7 |
| `data/analysis/weeks.csv.gz` | Week | 312 |
| `data/analysis/dropped_weeks.csv.gz` | Week | 2 |
| `data/analysis/collection_counts.csv.gz` | Statistic | 11 |
| `results/` | Numbers, tables, figures and estimates | |

The MiniLM labels can be used as they are or regenerated in about 5 hours with
`src/analysis/classify_panel_titles.py`.

### Regenerated by the full replication

| File | Source | Produced by |
|---|---|---|
| `data/raw/stackoverflow/stackoverflow_question_type_raw_*.csv` | SEDE (to 2023-05), API (2023-06 to 2024-12) | SQL in `sql/sede_question_type_raw_windows/` (written by `src/data/generate_sede_question_type_sql.py`); `src/data/fetch_stackoverflow_via_api.py` |
| `data/raw/stackoverflow/stackoverflow_tag_week.csv` | SEDE | `sql/stackoverflow_tag_week_top100.sql` |
| `data/raw/stackoverflow/stackoverflow_post_complexity_tag_week_{2018_2020,2021_2023,2024_2026}.csv` | SEDE | `sql/sede_windows/stackoverflow_post_complexity_tag_week_window.sql` |
| `data/raw/stackoverflow/stackoverflow_question_validation_pool.csv` | SEDE | `sql/stackoverflow_question_validation_pool.sql` |
| `data/raw/stackoverflow/api_2025_full_100tags/` | API | `src/data/fetch_stackoverflow_2025_full_100tags.py` |
| `data/processed/ai_answerability_index.csv` | Derived | `src/data/prepare_answerability_index.py` |
| `data/processed/stackoverflow_2025_clean_question_tag.csv` | Derived | `src/data/build_stackoverflow_2025_clean.py` |
| `data/processed/_validation/stackoverflow_question_type_raw_2023-05-02_2023-05-03.csv` | API | `src/data/fetch_stackoverflow_via_api.py --start 2023-05-02 --end 2023-05-03` |

## Computational requirements

### Software requirements

- Python 3.14.4. `requirements.txt` pins every Python package, direct and indirect, at
  the exact version that produced the results.
- R 4.6.1 with the package HonestDiD 0.2.8, called by `src/pipeline/honestdid.R`.
  `install_r_packages.R` installs HonestDiD and every package it depends on at the exact
  versions that produced the results, from the 2026-11-04 snapshot of Posit Package
  Manager, and stops if R is not 4.6.1 or if any version differs. On Windows it installs
  binary packages; on other systems it builds them from source, which needs the compilers
  and system libraries those packages require. The pipeline finds `Rscript` on the path
  or through the environment variable `RSCRIPT`, and R finds the packages through
  `R_LIBS`.
- A LaTeX distribution only if the article is compiled; the article
  sources are not part of this package.

### Controlled randomness

Every random draw uses the seed `SEED = 20260601` defined in
`src/pipeline/common.py`: the Kline–Santos score bootstrap, the random
allocation benchmark of the monitoring procedure, and the samples drawn
for the classifier and embedding validations.

`run_pipeline.py` also sets the environment variable `PYTHONHASHSEED`. pyfixest keeps the
fixed effects of a formula in a Python set, so the hash seed sets the order in which they
are demeaned. Other orders change no reported number and move unrounded coefficients,
standard errors and p-values by less than 1e-7.

### Memory, runtime and storage

Reference machine: Windows 11 laptop, Intel Core i5-1035G1, 20 GB of RAM, no GPU.

| Task | Time | Peak memory |
|---|---|---|
| Quick replication, `python run_pipeline.py --from-analysis-data` | 50 to 95 minutes | 5.6 GB in Python, 1.3 GB in R |
| Full replication: `python run_pipeline.py` | 75 to 100 minutes | 5.7 GB in Python, 1.3 GB in R |
| Full replication: MiniLM labels, `src.analysis.classify_panel_titles` with four workers | About 5 hours | |
| Full replication: SEDE queries, run by hand, one per four-day window | Several days | |

A machine with 8 GB of RAM should suffice and 12 GB is recommended; the package has been
tested only on the reference machine. Python and R do not peak at the same time. Do not
run other memory-heavy jobs at the same time.

Storage: `data/analysis/` takes 78 MB. The full replication adds about 2 GB for the raw
Stack Overflow files and about 0.5 GB for the embedding models, which
`sentence-transformers` caches.

## Description of the code

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Runs every stage in order and writes `results/`. |
| `src/pipeline/analysis_data.py` | Writes `data/analysis/` from the raw files and reads it. |
| `src/pipeline/data.py` | Loads the raw files, builds Monday-to-Sunday weeks, the question-type classification, the support-post funnel and the cells. |
| `src/pipeline/estimates.py` | PPML triple difference, funnel, vote groups, event study, HonestDiD, placebos, trends, heterogeneity, magnitudes, score bootstrap. |
| `src/pipeline/extension.py` | 2025 extension and the monitoring procedure. |
| `src/pipeline/validation.py` | Classifier agreement, embedding and revealed answerability, continuous substitutability, keyword and MiniLM sensitivity, survivor composition. |
| `src/pipeline/report.py` | Tables and figures. |
| `src/pipeline/honestdid.R` | HonestDiD sensitivity analysis. |
| `install_r_packages.R` | Installs the R packages at the versions that produced the results. |
| `src/pipeline/common.py` | Shared constants, estimators and the macro registry. |
| `src/data/` | Data acquisition and preparation scripts listed in the dataset list. `prepare_stackoverflow_question_type_raw.py` holds the question-type rule. |
| `src/analysis/classify_panel_titles.py` | MiniLM nearest-centroid label of every title in the main panel. |
| `sql/` | SEDE queries (see `sql/README.md` and `docs/stackoverflow_sede_download.md`). |

## Instructions to replicators

### Environment, for both routes

`python` must be Python 3.14.4 and R must be 4.6.1.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows, Git Bash; on Unix: source .venv/bin/activate
pip install -r requirements.txt
```

In cmd the second line is `.venv\Scripts\activate.bat`. In PowerShell it is
`.venv\Scripts\Activate.ps1`, which needs an execution policy that allows local scripts.

Then install the R packages into a new library in the package folder:

```bash
mkdir .Rlib
export R_LIBS="$PWD/.Rlib"
Rscript install_r_packages.R
```

In cmd the second line is `set R_LIBS=%CD%\.Rlib`. In PowerShell it is
`$env:R_LIBS = "$PWD\.Rlib"`. `R_LIBS` must also be set in the shell that runs the
pipeline. If `Rscript` is not on the path, call it by its full path, for example
`"C:/Program Files/R/R-4.6.1/bin/Rscript.exe"`, and set the environment variable `RSCRIPT`
to that path.

### 1. Quick replication from `data/analysis/`

```bash
python run_pipeline.py --from-analysis-data
```

It prints one line per stage and ends with `[done]`. It rewrites `results/`. On the
reference machine the rewritten files are identical to the committed ones.

### 2. Full replication from the sources

The data of the article were collected in May and June 2026, and a new download can
differ from them (see "Collection").

1. Obtain the SEDE extracts following `docs/stackoverflow_sede_download.md`
   and save them under `data/raw/stackoverflow/` with the names in the
   dataset list.
2. Register a free Stack Exchange API key and set the environment variable
   `STACKEXCHANGE_KEY` to it. Then fetch the API windows:

   ```bash
   python -m src.data.fetch_stackoverflow_via_api --start 2023-06-01 --end 2025-01-01
   python -m src.data.fetch_stackoverflow_via_api --start 2023-05-02 --end 2023-05-03 --days 1 --out-dir data/processed/_validation
   python -m src.data.fetch_stackoverflow_2025_full_100tags
   ```

   Question–tag pairs that appear in both a SEDE and an API file are
   counted once (the pipeline drops duplicate question–tag pairs).
3. Build the derived inputs:

   ```bash
   python -m src.data.prepare_answerability_index
   python -m src.data.build_stackoverflow_2025_clean
   python -m src.analysis.classify_panel_titles --workers 4
   ```

   The last command embeds about 5 million titles on the CPU; it resumes where it stopped
   if interrupted, and it can be skipped to use the labels in `data/analysis/`.
4. Run the analysis, which rewrites `data/analysis/` and `results/`:

   ```bash
   python run_pipeline.py
   ```

## List of tables and figures

Every table and figure comes from the pipeline, by either route. Files are in
`results/tables/` and `results/figures/`, and the functions are in
`src/pipeline/report.py`.

| Article | File | Written by |
|---|---|---|
| Table 1: dataset summary | `table_dataset_summary.tex` | `table_dataset` |
| Table 2: 2025 extension | `table_extension_2025.tex` | `table_extension` |
| Table 3: support-post funnel | `table_reusable_funnel.tex` | `table_funnel` |
| Table 4: inference for the central coefficients | `table_inference.tex` | `table_inference` |
| Table 5: literature positioning | `table_literature_positioning.tex` | `table_literature` |
| Table 6: value of information in allocation | `table_vtd_voi.tex` | `table_voi` |
| Figure 1: weekly volume by substitutability | `macro_substitutable_vs_not_log.png` | `figures` |
| Figure 2: event study | `event_study_ddd_question_type_quarterly.png` | `figures` |
| Figure 3: value disproportionality | `fig_vtd_monitor.pdf` | `figures` |
| Table A1: HonestDiD | `table_honestdid.tex` | `table_honestdid` |
| Table A2: identification diagnostics, funnel-qualified | `table_funnel_identification.tex` | `table_funnel_identification` |
| Table A3: HonestDiD, two estimators | `table_honestdid_estimators.tex` | `table_honestdid_estimators` |
| Table A4: robustness | `table_robustness.tex` | `table_robustness` |
| Table A5: score bins | `table_score_bins.tex` | `supplementary_tables` |
| Table A6: voting-normalized thresholds | `table_voting_normalised_highvalue.tex` | `supplementary_tables` |
| Table A7: log(1+Y) robustness | `table_log_robustness.tex` | `supplementary_tables` |
| Table A8: funnel by question type | `table_funnel_heterogeneity_by_qtype.tex` | `supplementary_tables` |
| Table A9: resolvability and quality | `table_resolvability_vs_quality.tex` | `supplementary_tables` |
| Table A10: pre/post means by tag-index group | `table_prepost_resolvability_by_ai_group.tex` | `supplementary_tables` |
| Table A11: embedding answerability | `table_embedding_validation.tex` | `supplementary_tables` |
| Table A12: embedding partial regressions | `table_embedding_partial_regression.tex` | `supplementary_tables` |
| Table A13: revealed answerability | `table_revealed_answerability.tex` | `supplementary_tables` |
| Table A14: AUC by stratum | `table_auc_stratum.tex` | `supplementary_tables` |
| Table A15: question-type rule | `table_classifier_rule.tex` | `table_classifier_rule` |
| Table A16: leave-one-keyword-out | `table_keyword_sensitivity.tex` | `table_keyword_sensitivity` |
| Table A17: MiniLM labels on the panel | `table_minilm_panel.tex` | `table_minilm_panel` |
| Table A18: other statistics quoted in the text | `table_text_estimates.tex` | `table_text_estimates` |
| Figure A1: event study, funnel-qualified | `fig_event_study_funnel.png` | `figures` |
| Figure A2: specification curve | `fig_specification_curve.png` | `figures` |
| Figure A3: rolling placebos | `fig_rolling_placebo_distribution.png` | `figures` |
| Figure A4: tag-level change against the index | `binscatter_ai_effect.png` | `figures` |
| Figure A5: funnel by question type | `fig_funnel_heterogeneity_by_qtype.pdf` | `figures` |
| Figure A6: resolvability coefficients | `fig_resolvability_coefficients.pdf` | `figures` |
| Figure A7: descriptive trend counterfactual | `economic_magnitude_counterfactual.png` | `figures` |

## References

- Stack Exchange, Inc. Stack Exchange Data Explorer. <https://data.stackexchange.com>.
- Stack Exchange, Inc. Stack Exchange API v2.3. <https://api.stackexchange.com/docs>.
- Rambachan, A., and J. Roth (2023). A more credible approach to parallel trends. *Review of Economic Studies* 90(5): 2555–2591.
- Kline, P., and A. Santos (2012). A score based approach to wild bootstrap inference. *Journal of Econometric Methods* 1(1): 23–41.
