"""Single run of the replication pipeline.

Without arguments it builds data/analysis/ from the raw files and then runs the
analysis. With --from-analysis-data it starts from data/analysis/, which is part
of this package, so nothing is downloaded. Either way it runs every estimate
reported in the article and its online resource, and writes
    results/numbers.tex      LaTeX macros for every number in the text
    results/tables/*.tex     tables (cells are macros from numbers.tex)
    results/figures/*        figures
    results/estimates/*.csv  unrounded estimates

Requirements: the Python packages in requirements.txt and R with the
HonestDiD package (set RSCRIPT and, if needed, R_LIBS).
Usage: python run_pipeline.py [--from-analysis-data]
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time

# pyfixest keeps the fixed effects of a formula in a set of strings, so the hash seed
# sets the order in which they are demeaned. This value gives the committed results.
HASH_SEED = "2"
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != HASH_SEED:
    os.environ["PYTHONHASHSEED"] = HASH_SEED
    sys.exit(subprocess.call([sys.executable, *sys.argv]))

import pandas as pd

from src.pipeline import analysis_data, estimates, extension, report, validation
from src.pipeline.common import ESTIMATES, FIGURES, RESULTS, TABLES, Est, Registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the replication pipeline.")
    parser.add_argument("--from-analysis-data", action="store_true",
                        help="start from data/analysis/ instead of the raw files; nothing is downloaded")
    args = parser.parse_args()
    for d in (RESULTS, TABLES, FIGURES, ESTIMATES):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not args.from_analysis_data:
        analysis_data.write_from_raw()
        gc.collect()
    D = analysis_data.load()
    print(f"[data] {len(D.main):,} cells main, {len(D.ext):,} cells extended ({time.time() - t0:.0f}s)", flush=True)

    R, out = Registry(), {"D": D}
    steps = [estimates.descriptives, estimates.source_validation, estimates.baseline, estimates.funnel, estimates.selenium_gap,
             estimates.index_without_maturity, estimates.score_bins,
             estimates.dynamics, estimates.honestdid, estimates.funnel_identification,
             estimates.beta_weighted_sensitivity, estimates.placebos, estimates.matched,
             estimates.spec_curve, estimates.trend_controls, estimates.trend_extrapolation,
             estimates.late_confounders, estimates.heterogeneity,
             estimates.magnitude, extension.extension, extension.monitor,
             validation.classifier_agreement, validation.embedding_answerability,
             validation.revealed, validation.continuous_substitutability, validation.keyword_sensitivity,
             validation.minilm_panel, validation.mechanisms]
    for step in steps:
        t = time.time()
        step(D, R, out)
        gc.collect()                   # model fits hold reference cycles; free them between steps
        print(f"[{step.__module__.split('.')[-1]}.{step.__name__}] {time.time() - t:.0f}s", flush=True)

    report.table_dataset(R)
    report.table_extension(R, out)
    report.table_funnel(R, out)
    report.table_inference(R)
    report.table_voi(R, out)
    report.table_literature()
    report.table_robustness(R, out)
    report.table_honestdid(R, out)
    report.table_funnel_identification(R, out)
    report.table_honestdid_estimators()
    report.table_text_estimates()
    report.table_classifier_rule(R)
    report.table_keyword_sensitivity(R, out)
    report.table_minilm_panel()
    report.supplementary_tables(R, out)
    report.figures(D, R, out)

    for k, v in out.items():
        if isinstance(v, pd.DataFrame):
            # question titles are Stack Overflow user content and are not redistributed
            v.drop(columns=["title"], errors="ignore").to_csv(ESTIMATES / f"{k}.csv", index=False)
        elif isinstance(v, Est):
            pd.DataFrame([v.row()]).to_csv(ESTIMATES / f"{k}.csv", index=False)
    R.write(RESULTS / "numbers.tex", ESTIMATES / "values.csv")
    print(f"[done] {len(R.vals)} macros, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
