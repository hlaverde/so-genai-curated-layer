"""LaTeX tables (every cell is a macro from results/numbers.tex) and figures."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.prepare_stackoverflow_question_type_raw import (  # noqa: E402
    NON_SUBSTITUTABLE, RULES, ZERO_WIDTH_PAT)

from .common import (AI, AI_VARIANTS, CHATGPT, FIGURES, TABLES,  # noqa: E402
                     Registry)
from .data import Data  # noqa: E402
from .estimates import QTYPES  # noqa: E402

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def letters(s: str) -> str:
    s = re.sub(r"\d", lambda m: " " + DIGITS[int(m.group())] + " ", str(s))
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", s) if w)


class T:
    """Registers each table cell as a macro and returns its reference."""

    def __init__(self, R: Registry, prefix: str):
        self.R, self.p = R, "T" + prefix

    def __call__(self, key: str, value, kind: str) -> str:
        name = self.p + letters(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "---"
        self.R.add(name, value, kind)
        return f"\\R{name}"


def write(name: str, lines: list[str]) -> None:
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def est_cells(t: T, key: str, row, n=True, signed=False, ci=False) -> list[str]:
    c = [t(f"{key}B", row["beta"], "scoef" if signed else "coef"),
         t(f"{key}SE", row["se"], "coef"), t(f"{key}P", row["p"], "pv")]
    if ci:
        c.append(f"[{t(f'{key}Lo', row['ci_low'], 'coef')}, {t(f'{key}Hi', row['ci_high'], 'coef')}]")
    if n:
        c.append(t(f"{key}N", row["n"], "int"))
    return c


def date_range(r) -> str:
    return f"{pd.Timestamp(r['first_week']):%Y-%m-%d} to {pd.Timestamp(r['last_day']):%Y-%m-%d}"


# --------------------------------------------------------------------------
# main-text tables
# --------------------------------------------------------------------------
def table_dataset(R: Registry) -> None:
    v = lambda k: f"\\R{k}{{}}"
    write("table_dataset_summary", [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Summary of the dataset and its construction. All quantities are derived from public "
        r"sources; the unit of analysis is the tag--week--question-type cell, and weeks run Monday to Sunday.}",
        r"\label{tab:dataset_summary}", r"\small", r"\begin{tabular}{ll}", r"\toprule",
        r"Component & Description \\", r"\midrule",
        r"Platform & Stack Overflow (public programming Q\&A) \\",
        r"Sources & Stack Exchange Data Explorer (through 2023-05); \\",
        r"        & Stack Exchange API v2.3 (from 2023-06) \\",
        f"Main window & {v('WeekFirst')} to {v('MainLastDay')} ({v('NWeeksMain')} complete weeks) \\\\",
        f"Extension & through {v('ExtLastDay')} ({v('NWeeksExt')} complete weeks, all {v('NTags')} tags) \\\\",
        r"Unit of analysis & Tag $\times$ week $\times$ question-type cell \\",
        f"Tags & {v('NTags')} top tags fixed by pre-ChatGPT volume \\\\",
        f"Question--tag pairs & {v('NPairsMain')} (main window) \\\\",
        f"Unique questions & {v('NUniqueMain')} (main window); 2025: {v('NCollectedTwentyFive')} collected, \\\\",
        f"        & {v('NUniqueTwentyFive')} in complete weeks \\\\",
        r"        & (the 2025 days of the last main-window week are in both) \\",
        f"Panel cells & {v('NCellsMain')} (main); {v('NCellsExt')} (with 2025) \\\\",
        r"Question types & short\_code, long\_code, how\_to, \\",
        r"        & debugging\_simple, other\_conceptual, \\",
        r"        & version\_environment\_specific, advanced\_architecture \\",
        r"Main outcome & Question--tag pairs per cell (count; PPML) \\",
        r"Main treatment & Pre-ChatGPT tag index (resolution and activity) \\",
        r"Value measure & Net-score bins and the support-post funnel \\",
        r"Reproducibility & Public scripts and derived data; raw content \\",
        r"        & not redistributed (CC BY-SA 4.0) \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def table_extension(R: Registry, out: dict) -> None:
    t = T(R, "Ext")
    es = out["es_ext"]
    post = es[(es["bin"] >= 0) & es["complete"]].sort_values("bin")
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{2025 extension on the full \RNTags-tag panel. Panel A re-estimates eq.~(\ref{eq:ddd}) "
             r"on the portion of the extended panel that coincides with the main window, on the full "
             r"extended panel, on the extended panel excluding advanced\_architecture cells, and on "
             r"funnel-qualified posts in the extended panel. Panel B "
             r"reports the post-period event-study coefficients of the extended panel for every complete "
             r"13-week bin (event time and dates); they are estimated on the full extended panel and so "
             r"differ slightly from the main-window coefficients of Figure~\ref{fig:event_study}. Panel C reports the persistence of the tag-level decline "
             r"in upvoted posts from 2024 into 2025.}",
             r"\label{tab:extension_2025}", r"\small", r"\begin{tabular}{lrr}", r"\toprule",
             r"\multicolumn{3}{l}{\emph{Panel A. Triple difference, full \RNTags-tag panel}} \\", r"\midrule",
             r" & $\hat\beta_{\text{DDD}}$ & $p$ \\", r"\midrule",
             f"Main-window portion ({R.vals['WeekFirst'][0]} to {R.vals['MainLastDay'][0]}) & \\RGateB & \\RGatePv \\\\",
             f"Full extended panel (to {R.vals['ExtLastDay'][0]}) & \\RExtFullB & \\RExtFullPv \\\\",
             r"Extended panel, excluding advanced\_architecture cells & \RExtNoAAB & \RExtNoAAPv \\",
             r"Extended panel, funnel-qualified posts & \RExtFunQualB & \RExtFunQualPv \\",
             r"\midrule",
             r"\multicolumn{3}{l}{\emph{Panel B. Event-study coefficients, complete post-period bins}} \\",
             r"\midrule", r"Event time & Dates & Coefficient \\", r"\midrule"]
    for _, r in post.iterrows():
        k = int(r["bin"])
        lines.append(f"T$+${k} & {date_range(r)} & {t(f'Bin{k}B', r['beta'], 'coef')} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{3}{l}{\emph{Panel C. Monitor persistence on 2025 data}} \\", r"\midrule",
              r"Spearman (decline in upvoted posts, 2024 and 2025) & \RPersistRho & \RPersistP \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("table_extension_2025", lines)


def table_funnel(R: Registry, out: dict, name="table_reusable_funnel", key="Fun", src="funnel",
                 label="tab:reusable_funnel", caption=None) -> None:
    t = T(R, key)
    cap = caption or (r"Triple-difference (DDD) estimates across the observed support-post funnel, by PPML "
                      r"on $Y$, the cell count of question--tag pairs at each stage. Coefficients on "
                      r"$\text{AI}_t\cdot\text{Post}_w\cdot\text{Sub}_k$ with tag-clustered standard errors. "
                      r"The implied shortfall, in question--tag pairs, is "
                      r"$\sum_{\text{post}}(Y e^{-\hat\beta\,(\text{AI}-\text{AI}_{\text{ref}})\cdot\text{Sub}}-Y)$, "
                      r"where $\text{AI}_{\text{ref}}$ is the index of the reference tag, \RRefTag, the tag "
                      r"closest to the 10th percentile of the tag index. $\hat\gamma_3$ is the coefficient on "
                      r"$\text{Post}_w\cdot\text{Sub}_k$, the post-period change in the gap between substitutable "
                      r"and non-substitutable types within a tag at the zero of the tag index (its mean across "
                      r"tags), with its tag-clustered $p$-value.")
    lines = [r"\begin{table}[ht]", r"\centering", f"\\caption{{{cap}}}", f"\\label{{{label}}}", r"\small",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Funnel stage & $\hat\beta_{\text{DDD}}$ & SE & $p$ & 95\% CI & N cells & $\hat\gamma_3$ & $p(\hat\gamma_3)$ & "
             r"Implied shortfall (pairs) \\",
             r"\midrule"]
    for _, r in out[src].iterrows():
        k = letters(r["label"])[:24]
        lines.append(f"{r['label']} & " + " & ".join(est_cells(t, k, r, ci=True)) +
                     f" & {t(k + 'GThree', r['g3'], 'scoef')} & {t(k + 'GThreeP', r['g3_p'], 'pv')}"
                     f" & {t(k + 'Short', r['shortfall'], 'int')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    write(name, lines)


def table_inference(R: Registry) -> None:
    write("table_inference", [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Inference for the two central coefficients of eq.~(\ref{eq:ddd}), with the pre-ChatGPT "
        r"tag index and with the embedding-based answerability score as the treatment intensity. The tag "
        r"index is a mean of $z$-scores and is not re-standardized, so its coefficients are per unit of the "
        r"index; the embedding score is standardized, so its coefficients are per standard deviation, and "
        r"the two panels are on different scales. CR1: "
        r"standard errors clustered by tag; score bootstrap: Kline--Santos with \RKSReps{} Rademacher "
        r"draws and the null imposed; CR3: leave-one-tag-out jackknife. The last two columns re-estimate "
        r"the model without python and give the range of the \RNTags{} leave-one-tag-out estimates.}",
        r"\label{tab:inference}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Coefficient & $\hat\beta_{\text{DDD}}$ & 95\% CI (CR1) & $p$ CR1 & $p$ score bootstrap & "
        r"SE CR3 & $p$ CR3 & Without python & Leave-one-out range \\", r"\midrule",
        r"\multicolumn{9}{l}{\emph{Pre-ChatGPT tag index (resolution and activity)}} \\",
        r"Question volume & \RBaseB & [\RBaseLo, \RBaseHi] & \RBasePv & \RKSQPv & \RBaseCRThreeSE & "
        r"\RBaseCRThreePv & \RNoPythonQB & [\RLooAllQMin, \RLooAllQMax] \\",
        r"Funnel-qualified posts & \RFunQualB & [\RFunQualLo, \RFunQualHi] & \RFunQualPv & \RKSQualPv & "
        r"\RQualCRThreeSE & \RQualCRThreePv & \RNoPythonQualB & [\RLooAllQualMin, \RLooAllQualMax] \\",
        r"\multicolumn{9}{l}{\emph{Embedding-based answerability score}} \\",
        r"Question volume & \REmbDDDB & [\REmbDDDLo, \REmbDDDHi] & \REmbDDDPv & \RKSEmbQPv & "
        r"\REmbQCRThreeSE & \REmbQCRThreePv & \RNoPythonEmbQB & [\RLooAllEmbQMin, \RLooAllEmbQMax] \\",
        r"Funnel-qualified posts & \REmbQualB & [\REmbQualLo, \REmbQualHi] & \REmbQualPv & \RKSEmbQualPv & "
        r"\REmbQualCRThreeSE & \REmbQualCRThreePv & \RNoPythonEmbQualB & "
        r"[\RLooAllEmbQualMin, \RLooAllEmbQualMax] \\",
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])


def table_voi(R: Registry, out: dict) -> None:
    t = T(R, "VOI")
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Allocation-margin value of information on upvoted posts on the full \RNTags-tag panel "
             r"(out of sample). Upvoted posts are funnel-qualified posts with score $\geq1$. A manager allocates an "
             r"intervention budget $B$ (a fraction of total 2024 weekly activity) to protect the realized 2025 "
             r"flow of upvoted posts; covering a tag costs its 2024 weekly activity, and each rule ranks tags by "
             r"its signal per unit of cost. Declines are in weekly levels against the tag's pre-ChatGPT mean. "
             r"\textbf{Oracle} is the perfect-foresight benchmark, which ranks tags by their realized 2025 loss "
             r"per unit of cost; \textbf{Value-based} ranks by the observed 2024 decline in weekly upvoted posts; "
             r"\textbf{Naive} by the 2024 decline in weekly activity; \textbf{Random} averages random orders. "
             r"Cells are the share of the realized 2025 loss of upvoted posts captured; the regret of a rule is "
             r"$(\text{Oracle}-\text{rule})/\text{Oracle}$, computed from unrounded shares.}",
             r"\label{tab:vtd_voi}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"$B$ & Oracle & Value-based & Naive & Random & Regret$_{\text{value}}$ & Regret$_{\text{naive}}$ \\",
             r"\midrule"]
    for _, r in out["voi"].iterrows():
        k = f"B{int(round(100 * r['budget']))}"
        lines.append(f"{t(k + 'Budget', r['budget'], 'pct0')}\\% & " + " & ".join(
            t(k + c, r[c], "num2") for c in
            ["oracle", "calibrated", "naive", "random", "regret_calibrated", "regret_naive"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("table_vtd_voi", lines)


def table_literature() -> None:
    write("table_literature_positioning", [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Positioning of this paper relative to recent empirical work on the post-ChatGPT shock to "
        r"programming Q\&A platforms. A check mark indicates that the cited study reports a substantive "
        r"empirical result on the corresponding margin. ``Task-level DDD'' refers to estimation via a triple "
        r"difference exploiting variation in task-level LLM substitutability; ``support-post funnel'' refers "
        r"to tracing the effect through support stages: answered, accepted, accepted and not closed, "
        r"accepted with non-negative score, and funnel-qualified (all of these gates at once).}",
        r"\label{tab:literature_positioning}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Study & \makecell{Quantity\\decline} & \makecell{Quality /\\composition} & "
        r"\makecell{Users /\\retention} & Answers & \makecell{Novelty /\\readability} & "
        r"\makecell{Task-level\\DDD} & \makecell{Support-post\\funnel} \\", r"\midrule",
        r"\citet{delriochanona2024large} & \checkmark & \checkmark & & & & & \\",
        r"\citet{burtch2024chatgpt} & \checkmark & \checkmark & \checkmark & & & & \\",
        r"\citet{xue2026chatgpt} & \checkmark & \checkmark & \checkmark & \checkmark & & & \\",
        r"\citet{borwankar2023unraveling} & \checkmark & \checkmark & \checkmark & \checkmark & & & \\",
        r"\citet{shanqiu2025isr} & \checkmark & \checkmark & & \checkmark & & & \\",
        r"\citet{quinngutt2024heterogeneous} & \checkmark & \checkmark & \checkmark & & \checkmark & & \\",
        r"\textbf{This paper} & \checkmark & \checkmark & & & & \checkmark & \checkmark \\",
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])


# --------------------------------------------------------------------------
# supplementary tables
# --------------------------------------------------------------------------
def table_robustness(R: Registry, out: dict) -> None:
    t = T(R, "Rob")
    rows: list[tuple[str, str, object, bool]] = []

    def row(label, key, e, n=True, signed=False):
        r = e.row() if hasattr(e, "row") else e
        return f"  {label} & " + " & ".join(est_cells(t, key, r, n=n, signed=signed)) + r" \\"

    base = out["base"]
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Robustness of $\hat\beta_{\text{DDD}}$. Unless stated otherwise each row re-estimates "
         r"eq.~(\ref{eq:ddd}) by PPML on the main panel with the pre-ChatGPT tag index, tag $\times$ "
         r"question-type and week fixed effects, and standard errors clustered by tag (CR1).}",
         r"\label{tab:robustness}", r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{lrrrr}", r"\toprule", r"Specification & $\hat\beta_{\text{DDD}}$ & SE & $p$ & N cells \\",
         r"\midrule", r"\multicolumn{5}{l}{\textit{A. Baseline and treatment-measure variants}} \\",
         row("Tag index (baseline)", "Base", base)]
    for k, lab in AI_VARIANTS.items():
        L.append(row(f"Variant: {lab} index", letters(lab), out["variants"][k]))
    L.append(row("Variant: tag index without tag maturity", "NoMaturity", out["nomat_q"]))
    L.append(row("Embedding-based answerability score", "Emb", out["embed_ddd"]))
    L += [r"\midrule", r"\multicolumn{5}{l}{\textit{B. Control group and sample restrictions}} \\",
          row(r"Excluding advanced\_architecture cells", "NoAA", out["noaa"]),
          row(r"advanced\_architecture recoded as substitutable", "Recode",
              out["control_fragility"].iloc[6])]
    for n in (50, 75):
        L.append(row(f"Top-{n} tags by pre-period volume", f"Top{n}", out[f"top{n}"]))
    m = out["matched"]
    L.append(row(f"Pre-slope within one IQR of the median ({int(m.iloc[4]['n_tags'])} tags)", "Band", m.iloc[4]))
    for i, lab in [(0, "Pre-slope quartile 1 (steepest decline)"), (3, "Pre-slope quartile 4 (steepest growth)")]:
        L.append(row(lab, f"SlopeQ{i + 1}", m.iloc[i], signed=True))
    for n, e in out["loo"].items():
        L.append(row(f"Excluding the top-{n} tag{'s' if n > 1 else ''} by pre-period volume", f"Drop{n}", e))
    L += [r"\midrule", r"\multicolumn{5}{l}{\textit{C. Outcomes and estimators}} \\",
          row("Outcome: unique users", "Users", out["users"]),
          row(r"Fractional counts ($\sum_q 1/k_q$, $k_q$ = panel tags of question $q$)", "Frac", out["frac"]),
          row(r"OLS on $\log(1+\text{questions})$", "LogBase", out["log_base"]),
          row("Two-way clustering (tag and week)", "TwoWay", out["twoway"]),
          row("CR3 (leave-one-tag-out jackknife) standard errors", "CRThree", out["cr3"]),
          r"  Kline--Santos score bootstrap (\RKSReps{} Rademacher draws, null imposed) & "
          r"\RBaseB & --- & \RKSQPv & --- \\",
          r"\midrule", r"\multicolumn{5}{l}{\textit{D. Pre-period placebos (pre-ChatGPT sample)}} \\"]
    for _, r in out["placebos"].iterrows():
        L.append(row(f"Placebo cutoff {r['cutoff']}", f"Plac{r['cutoff']}", r, signed=True))
    L += [r"\midrule", r"\multicolumn{5}{l}{\textit{E. Joint model with ChatGPT and Copilot cutoffs}} \\",
          row(r"AI$\times$Sub$\times$Post-ChatGPT", "JointGPT", out["joint"][0]),
          row(r"AI$\times$Sub$\times$Post-Copilot", "JointCop", out["joint"][1], signed=True),
          r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    write("table_robustness", L)


def table_honestdid(R: Registry, out: dict) -> None:
    t = T(R, "HD")
    L = [r"\begin{table}[ht]", r"\centering",
         r"\caption{HonestDiD robust 95\% confidence sets for $\theta$, the average of the complete "
         r"post-period event-study coefficients, under the relative-magnitudes restriction "
         r"$\Delta^{\text{RM}}(\bar M)$ (C-LF) and the smoothness restriction $\Delta^{\text{SD}}(M)$ "
         r"(FLCI), with the full cluster-robust event-study VCV. $\bar M$ bounds post-period deviations "
         r"from parallel trends relative to the largest change between consecutive pre-period bins; $M$ bounds the change in "
         r"the slope of the differential trend between consecutive 13-week bins. The C-LF endpoints are "
         r"located on a $\theta$ grid of step \RHDThetaGridStep{}; both sets use the normal reference of the package.}",
         r"\label{tab:honestdid}", r"\small", r"\begin{tabular}{rrl}", r"\toprule",
         r"\multicolumn{3}{l}{\emph{Panel A. Relative magnitudes}} \\", r"\midrule",
         r"$\bar M$ & Robust 95\% CI & Contains 0? \\", r"\midrule"]
    for _, r in out["honestdid"].iterrows():
        k = f"M{int(round(100 * r['Mbar']))}"
        has0 = r["lb"] <= 0 <= r["ub"]
        L.append(f"{t(k + 'Mbar', r['Mbar'], 'num2')} & $[{t(k + 'Lo', r['lb'], 'coef')}, "
                 f"{t(k + 'Hi', r['ub'], 'scoef')}]$ & {'Yes' if has0 else 'No'} \\\\")
    L += [r"\midrule", r"\multicolumn{3}{l}{\emph{Panel B. Smoothness}} \\", r"\midrule",
          r"$M$ & Robust 95\% CI & Contains 0? \\", r"\midrule"]
    for _, r in out["honestdid_sd"].iterrows():
        k = f"SD{int(round(10000 * r['M']))}"
        has0 = r["lb"] <= 0 <= r["ub"]
        L.append(f"{t(k + 'M', r['M'], 'coef4')} & $[{t(k + 'Lo', r['lb'], 'coef')}, "
                 f"{t(k + 'Hi', r['ub'], 'scoef')}]$ & {'Yes' if has0 else 'No'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("table_honestdid", L)


def table_honestdid_estimators() -> None:
    write("table_honestdid_estimators", [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{HonestDiD for two estimators of each outcome, both measured against bin $-1$ on the complete "
        r"event-study bins: $\theta$, the equal-weighted average of the post-period coefficients, and "
        r"$\theta_\beta$, the same coefficients with the post-period weights that approximately reproduce $\beta_{\text{DDD}}$, "
        r"rescaled to sum to one over the complete bins. $\beta_{\text{DDD}}$ itself also weights the pre-period "
        r"coefficients, so it is not equal to $\theta_\beta$. A breakdown value is the first value whose robust "
        r"confidence set contains zero, on a grid of step \RHDBreakdownStep{} for $\bar M$ (with a $\theta$ grid of "
        r"step \RHDThetaGridStep{}) and \RHDSDBreakdownStep{} for $M$. The 95\% CI of the estimates uses $t(\RClusterDf)$; the "
        r"robust sets use the normal reference of the package.}",
        r"\label{tab:honestdid_estimators}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"& \multicolumn{2}{c}{Question volume} & \multicolumn{2}{c}{Funnel-qualified} \\",
        r"& $\theta$ & $\theta_\beta$ & $\theta$ & $\theta_\beta$ \\", r"\midrule",
        r"Estimate & \RHDTheta & \RHDBTheta & \RQualHDTheta & \RQualHDBTheta \\",
        r"95\% CI & [\RHDOrigLo, \RHDOrigHi] & [\RHDBOrigLo, \RHDBOrigHi] & [\RQualHDOrigLo, \RQualHDOrigHi] & "
        r"[\RQualHDBOrigLo, \RQualHDBOrigHi] \\",
        r"$\Delta^{\text{RM}}$ breakdown $\bar M$ & \RHDBreakdown & \RHDBBreakdown & \RQualHDBreakdown & "
        r"\RQualHDBBreakdown \\",
        r"FLCI, linear trend ($M=0$) & \RHDSDCenterZero & \RHDBSDCenterZero & \RQualHDSDCenterZero & "
        r"\RQualHDBSDCenterZero \\",
        r"95\% CI & [\RHDSDLoZero, \RHDSDHiZero] & [\RHDBSDLoZero, \RHDBSDHiZero] & "
        r"[\RQualHDSDLoZero, \RQualHDSDHiZero] & [\RQualHDBSDLoZero, \RQualHDBSDHiZero] \\",
        r"$\Delta^{\text{SD}}$ breakdown $M$ & \RHDSDBreakdown & \RHDBSDBreakdown & \RQualHDSDBreakdown & "
        r"\RQualHDBSDBreakdown \\",
        r"$\beta_{\text{DDD}}$ approximately reproduced from all the weights & & \RHDBRepro & & \RQualHDBRepro \\",
        r"Post-period weight on $T{+}0$; $T{+}7$ (\%) & & \RHDBWeightFirst; \RHDBWeightLast & & "
        r"\RQualHDBWeightFirst; \RQualHDBWeightLast \\",
        r"Weight on the partial bin $T{+}8$ (\%) & & \RHDBPartialShare & & \RQualHDBPartialShare \\",
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])


def table_text_estimates() -> None:
    """Estimates and statistics quoted in the text, in particular those no other table shows."""
    def est(label, nm, p=None):
        return rf"{label} & \R{nm}B & \R{nm}SE & \R{p or nm + 'Pv'} \\"

    def ci(label, nm, lo=None, hi=None, p=""):
        return rf"{label} & \R{nm} & [\R{lo or nm + 'Lo'}, \R{hi or nm + 'Hi'}] & {p} \\"

    def val(label, *nm):
        return f"{label} & " + "; ".join(rf"\R{n}" for n in nm) + r" & & \\"

    def head(t):
        return [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{t}}}}} \\", r"\midrule"]

    L = [r"{\footnotesize", r"\begin{longtable}{p{6.2cm}rrr}",
         r"\caption{Estimates and statistics quoted in the text, in particular those that appear in no other table.  "
         r"Unless stated otherwise, $\hat\beta$ is the coefficient of eq.~(\ref{eq:ddd}) by PPML with "
         r"standard errors clustered by tag.}\label{tab:text_estimates} \\", r"\toprule",
         r"Statistic & Estimate & SE or 95\% CI & $p$ \\", r"\midrule", r"\endfirsthead", r"\toprule",
         r"Statistic & Estimate & SE or 95\% CI & $p$ \\", r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    L += [rf"\multicolumn{{4}}{{l}}{{\emph{{A. Windows and control definitions}}}} \\", r"\midrule",
          est(r"Donut (strike window dropped)", "Donut"), est(r"Window ending before the strike", "PreStrike"),
          est(r"Sole control: short code", "SoleShort"), est(r"Sole control: how-to", "SoleHowTo"),
          est(r"Sole control: long code", "SoleLong"), est(r"Sole control: debugging", "SoleDebug"),
          est(r"Sole control: other conceptual", "SoleOther"), est(r"Sole control: version-environment-specific", "SoleVES"),
          est(r"Sole control: advanced architecture", "SoleAA"),
          ci(r"Funnel-qualified, advanced architecture excluded", "NoAAQualB", "NoAAQualLo", "NoAAQualHi", r"\RNoAAQualPv"),
          ci(r"Version-environment-specific, AI$\times$Post (MDE \RHetVESMDE)", "HetVESB", "HetVESLo", "HetVESHi", r"\RHetVESPv"),
          ci(r"Advanced architecture, AI$\times$Post", "HetAAB", "HetAALo", "HetAAHi", r"\RHetAAPv"),
          est(r"Short code, AI$\times$Post", "HetShort"), est(r"How-to, AI$\times$Post", "HetHowTo"), est(r"Long code, AI$\times$Post", "HetLong"),
          est(r"Debugging, AI$\times$Post", "HetDebug"), est(r"Other conceptual, AI$\times$Post", "HetOther"), est(r"Pre-slope quartile 2", "SlopeQTwo"),
          est(r"Pre-slope quartile 3", "SlopeQThree")]
    L += head("B. Equality between stages and the vote margin")
    L += [ci(r"Funnel-qualified minus questions (stacked)", "FunEqB", "FunEqLo", "FunEqHi", r"\RFunEqPv"),
          rf"Joint Wald test of the six stages, $\chi^2(\RFunJointDf)$ & \RFunJointChi & & \RFunJointPv \\",
          ci(r"Attenuation, $1-\beta_{\text{FQ}}/\beta_{\text{vol}}$, delta method (\%)", "FunAtten"),
          ci(r"Upvoted minus zero-score (stacked)", "VoteDiffB", "VoteDiffLo", "VoteDiffHi", r"\RVoteDiffPv"),
          rf"Upvoted posts, score bootstrap & \RVWHighB & & \RKSHighPv \\",
          val(r"Weighted average of the score bins: PPML; $\log(1+Y)$", "PPBinsWAvg", "LogBinsWAvg"),
          val(r"Funnel-qualified coefficient in $\log(1+Y)$", "LogFunQualB")]
    L += head("C. Event study and trends (question volume)")
    L += [est(r"Event study, $T{+}0$", "ESPostZero"), est(r"Event study, $T{+}1$", "ESPostOne"),
          est(r"Event study, $T{+}7$", "ESPostSeven"), val(r"Mean of the pre-period bins; post-period bins", "ESPreMean", "ESPostMean"),
          rf"Joint test of the pre-period bins, $\chi^2$ & \RESPreJointChi & & \RESPreJointP \\",
          rf"Differential pre-trend per year & \RPreTrendB & \RPreTrendSE & \\",
          est(r"Pre-period extrapolation, no trend", "TrendRef"), est(r"Pre-period extrapolation, linear", "TrendLin"),
          est(r"Pre-period extrapolation, quadratic", "TrendQuad"),
          ci(r"Pre-period curvature per 13-week bin (largest absolute bound \RPreCurvBound)", "PreCurv"),
          val(r"Raw second differences of the pre-period path, absolute value: minimum; maximum", "PreRawDiffMin", "PreRawDiffMax"),
          est(r"Linear trend fitted on the full sample", "LinTrend"), est(r"Quadratic trend fitted on the full sample", "QuadTrend"),
          val(r"Rolling placebos: number; minimum; maximum; mean", "RollN", "RollMin", "RollMax", "RollMean"),
          val(r"Specification curve: cells; negative; negative and significant", "SpecN", "SpecNeg", "SpecNegSig"),
          val(r"Consecutive post-period bins in the extended event study", "ExtESNPost")]
    L += head("D. Magnitudes")
    L += [val(r"10th and 90th percentiles of the tag index", "AIPTen", "AIPNinety"),
          ci(r"10th to 90th percentile, question volume (\%)", "PropQ"),
          ci(r"10th to 90th percentile, funnel-qualified posts (\%)", "PropQual"),
          ci(r"Shortfall relative to the reference tag, question--tag pairs", "ShortQ"),
          ci(r"Same, unique questions", "ShortQU"), ci(r"Same, funnel-qualified pairs", "ShortQual"),
          ci(r"Same, upvoted pairs", "ShortHV"),
          val(r"Volume to funnel-qualified shortfall; funnel-qualified share (\%)", "ShortRatio", "ShortQualShare"),
          val(r"Funnel-qualified flow per year; over five years", "QualFlowYear", "QualFlowFiveYears"),
          val(r"Share of question--tag pairs in the largest tag (\%)", "LargestTagShare"),
          val(r"Largest regret, tags ranked by their observed 2024 decline (\%)", "RegretCalibMax")]
    L += head("E. Mechanisms, monitor and measurement")
    L += [rf"Post, body length (characters) & \RCompBodyChars & & \RCompBodyPv \\", est(r"Post, accepted-answer share", "CompAccShare"),
          val(r"Post$\times$tag index, range of $p$", "CompAIPMin", "CompAIPMax"),
          est(r"Continuous substitutability score", "ContSub"), est(r"Joint model, ChatGPT cutoff", "JointGPT"),
          est(r"Joint model, Copilot cutoff", "JointCop"), rf"Tag-level $r$, volume change and tag index & \RTagCorrR & & \RTagCorrP \\",
          ci(r"Monitor: AUC of the activity decline (\RMonitorBootReps{} draws over tags)", "AUCAct"),
          ci(r"Monitor: AUC of the VtD ratio", "AUCVtD"), ci(r"Monitor: Spearman persistence", "PersistRho"),
          rf"Share of upvoted posts in the shortfall vs its size, Spearman & \RVtDRho & & \RVtDRhoP \\",
          val(r"Pooled ratio of Figure 3; tags above it; tags plotted", "VtDTheta", "VtDNAbove", "VtDNTags"),
          rf"Embedding score and tag index: $r$; $\rho$ & \REmbCorrR; \REmbCorrRho & & \REmbCorrP \\",
          val(r"Embedding score, titles per tag: maximum; minimum; median; total", "EmbPerTagMax", "EmbPerTagMin",
              "EmbPerTagMedian", "EmbN"),
          val(r"Revealed answerability and tag index: $r$; question-level AUC", "RevealR", "RevealAUC"),
          val(r"Classifier agreement, embeddings: $\kappa$ binary; seven-way", "KappaEmbBin", "KappaEmbSeven"),
          val(r"Classifier agreement, heuristic vs embeddings: $\kappa$ range", "KappaHeurMin", "KappaHeurMax"),
          val(r"Fleiss' $\kappa$: binary; seven-way", "FleissBin", "FleissSeven"),
          val(r"All three classifiers agree (\%); questions in the sample", "AgreeAll", "ValN")]
    L += head("F. Descriptive statistics")
    L += [val(r"Weekly question--tag pairs: pre; post; change (\%)", "WeeklyPre", "WeeklyPost", "WeeklyChg"),
          val(r"Change (\%): substitutable; non-substitutable", "WeeklyChgSub", "WeeklyChgNonSub"),
          val(r"Change (\%): first and fourth quartile of the tag index", "QuartChgOne", "QuartChgFour"),
          val(r"Funnel-qualified share of pairs (\%); upvoted and score $\geq5$ shares of them (\%)", "FunnelSurvive", "ShareGeOne", "ShareGeFive"),
          val(r"Weeks: pre-period; post-period", "NWeeksPre", "NWeeksPost"),
          val(r"Questions created in 2025: collected; in complete weeks", "NCollectedTwentyFive", "NUniqueTwentyFive"),
          val(r"Tag maturity (weeks): \RMaturityTag; the other \RMaturityNOther{} tags; $z$-score of \RMaturityTag",
              "MaturityWeeksTag", "MaturityWeeksOther", "MaturityZ"),
          val(r"Body-length threshold of the question types (characters)", "ClassBodyMax"),
          val(r"Keyword and MiniLM labels on the panel: agreement (\%); binary Cohen's $\kappa$", "MiniLMAgree", "MiniLMKappa"),
          val(r"Leave-one-keyword-out, question volume: lowest; highest coefficient", "LokoQMin", "LokoQMax"),
          val(r"Leave-one-keyword-out, funnel-qualified posts: lowest; highest coefficient", "LokoQualMin", "LokoQualMax"),
          val(r"Selenium: pairs in the backfill; longest gap in any other tag (days)", "SelBackfillPairs", "SelOtherGapDays"),
          val(r"Selenium gap weeks dropped: volume; funnel-qualified", "SelDropQB", "SelDropQualB"),
          val(r"Selenium backfill removed: volume; funnel-qualified", "SelNoBackQB", "SelNoBackQualB"),
          val(r"Selenium tag dropped: volume; funnel-qualified", "SelNoTagQB", "SelNoTagQualB"),
          val(r"Main panel (million): question--tag pairs; unique questions", "NPairsMainM", "NUniqueMainM"),
          val(r"Question--tag pairs (million): 2020; 2024", "PairsTwentyTwentyM", "PairsTwentyTwentyFourM"),
          val(r"Source validation, 2023-05-02: rows whose body differs", "ValBodyDiff"),
          val(r"Partial weeks excluded: first; last", "DropWeekStart", "DropWeekEnd"),
          val(r"Start of the tag-index window", "IndexWinStart"),
          val(r"Rolling placebos: first; last cutoff", "RollFirst", "RollLast"),
          val(r"Empty cells (\%): main panel; 2025 (of \RNCellsTwentyFive{} cells)", "CellsEmptyMain", "CellsEmptyTwentyFive"),
          val(r"Source validation, 2023-05-02: API rows; SEDE rows; titles escaped", "ValAPIRows", "ValSEDERows", "ValTitleEscaped"),
          val(r"Descriptive trend: slope per week; $R^2$", "TrendSlope", "TrendRsq"),
          val(r"Post-period pairs (million): counterfactual; observed; gap", "TrendCFM", "ObsPostM", "TrendShortM"),
          val(r"Break from the descriptive trend (\%)", "TrendBreakPct")]
    L += [r"\end{longtable}", r"}"]
    write("table_text_estimates", L)


def table_funnel_identification(R: Registry, out: dict) -> None:
    t = T(R, "QI")
    L = [r"\begin{table}[ht]", r"\centering",
         r"\caption{Identification diagnostics for the funnel-qualified count, with the specification used for "
         r"question volume: date placebos on the pre-ChatGPT sample (Panel A), the PPML event study "
         r"(Panel B; Figure~\ref{fig:event_study_funnel}) and HonestDiD bounds for $\theta$, the average of its "
         r"complete post-period coefficients (Panel C). A breakdown value is the first value whose robust confidence "
         r"set contains zero, on a grid of step \RHDBreakdownStep{} for $\bar M$ (with a $\theta$ grid of step "
         r"\RHDThetaGridStep{}) and \RHDSDBreakdownStep{} for $M$.}",
         r"\label{tab:funnel_identification}", r"\small", r"\begin{tabular}{lrrr}", r"\toprule",
         r"\multicolumn{4}{l}{\emph{Panel A. Date placebos}} \\", r"\midrule",
         r"Cutoff & $\hat\beta$ & SE & $p$ \\", r"\midrule"]
    for i, r in out["placebos_qual"].reset_index(drop=True).iterrows():
        k = f"Plac{DIGITS[i]}"
        L.append(f"{r['cutoff']} & {t(k + 'B', r['beta'], 'scoef')} & {t(k + 'SE', r['se'], 'coef')} & "
                 f"{t(k + 'P', r['p'], 'pv')} \\\\")
    L += [r"\midrule", r"\multicolumn{4}{l}{\emph{Panel B. Event study}} \\", r"\midrule",
          r"Mean of the complete pre-period bins & \RQualESPreMean & & \\",
          r"Mean of the complete post-period bins ($\theta$) & \RQualESPostMean & & \\",
          r"Range of the post-period bins & [\RQualESPostMin, \RQualESPostMax] & & \\",
          r"Joint test of the pre-period bins & & & \RQualESPreJointP \\",
          r"\midrule", r"\multicolumn{4}{l}{\emph{Panel C. HonestDiD}} \\", r"\midrule",
          r"$\theta$, 95\% CI & \RQualHDTheta & \multicolumn{2}{r}{[\RQualHDOrigLo, \RQualHDOrigHi]} \\",
          r"$\Delta^{\text{RM}}$ breakdown $\bar M$ & \RQualHDBreakdown & & \\",
          r"FLCI, linear trend ($M=0$) & \RQualHDSDCenterZero & \multicolumn{2}{r}{[\RQualHDSDLoZero, \RQualHDSDHiZero]} \\",
          r"$\Delta^{\text{SD}}$ breakdown $M$ & \RQualHDSDBreakdown & & \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("table_funnel_identification", L)


def table_simple(R, name, key, label, caption, header, df, cols, kinds, first, first_key=None):
    t = T(R, key)
    L = [r"\begin{table}[ht]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\small",
         r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{l" + "r" * len(cols) + "}",
         r"\toprule", header + r" \\", r"\midrule"]
    for i, r in df.reset_index(drop=True).iterrows():
        rk = letters(first_key(r) if first_key else r[first])[:28] or f"Row{DIGITS[i % 10]}"
        cells = []
        for c, k in zip(cols, kinds):
            if k == "ci":
                cells.append(f"[{t(rk + 'Lo', r['ci_low'], 'coef')}, {t(rk + 'Hi', r['ci_high'], 'coef')}]")
            else:
                cells.append(t(rk + letters(c), r[c], k))
        L.append(f"{r[first]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    write(name, L)


def supplementary_tables(R: Registry, out: dict) -> None:
    b = out["bins"].copy()
    b["label"] = ["score = 0", "score = 1", "score = 2", "score = 3", "score = 4", r"score $\geq$ 5"]
    table_simple(R, "table_score_bins", "Bins", "tab:score_bins",
                 r"Mutually exclusive score bins of funnel-qualified posts (answered, accepted, not closed, "
                 r"and non-negative in score). Each row re-estimates eq.~(\ref{eq:ddd}) by PPML with the bin "
                 r"count as the outcome; Holm $p$ adjusts for the six bins; MDE is the minimum detectable effect "
                 r"at 80\% power ($2.80\times$SE). Posts are question--tag pairs. $N$ is the number of cells in the "
                 r"estimation: tag-by-question-type series that are zero throughout the period are dropped.",
                 r"Bin & $\hat\beta$ & SE & $p$ & Holm $p$ & 95\% CI & MDE & Posts & $N$",
                 b, ["beta", "se", "p", "p_holm", "ci", "mde80", "posts", "n"],
                 ["coef", "coef", "pv", "pv", "ci", "num2", "int", "int"], "label")
    vn = out["voting"].copy()
    vn["label"] = [f"score $\\geq$ {k} (voting-normalized)" for k in vn["k"]]
    table_simple(R, "table_voting_normalised_highvalue", "VN", "tab:voting_normalised",
                 r"Funnel-qualified posts above each score threshold after rescaling post-period scores by each tag's "
                 r"pre/post ratio of mean absolute score (ratio clipped to $[0.5,3]$); PPML, as in "
                 r"eq.~(\ref{eq:ddd}).",
                 r"Outcome & $\hat\beta$ & SE & $p$ & 95\% CI & Posts", vn,
                 ["beta", "se", "p", "ci", "posts"], ["coef", "coef", "pv", "ci", "int"], "label")
    table_simple(R, "table_log_robustness", "LogR", "tab:log_robustness",
                 r"OLS on $\log(1+Y)$ as a robustness check on the PPML estimates. Each row re-estimates "
                 r"eq.~(\ref{eq:ddd}) for a funnel stage, one of the two vote groups of funnel-qualified "
                 r"posts, or one of the score bins of Table~\ref{tab:score_bins} (score 0 is the zero-score "
                 r"vote group), with the same fixed effects and tag clustering; the PPML columns repeat the main "
                 r"estimates.",
                 r"Outcome & PPML $\hat\beta$ & $p$ & $\log(1+Y)$ $\hat\beta$ & SE & $p$ & 95\% CI",
                 out["log_robust"], ["ppml_beta", "ppml_p", "beta", "se", "p", "ci"],
                 ["scoef", "pv", "scoef", "coef", "pv", "ci"], "label")
    het = out["het"].pivot(index="question_type", columns="outcome", values="beta")
    het = het.loc[[q for q, _ in QTYPES]]
    het = het[["questions", "answered", "accepted", "acc_nonclosed", "acc_nonneg", "funnel"]].reset_index()
    het["question_type"] = het["question_type"].str.replace("_", r"\_")
    table_simple(R, "table_funnel_heterogeneity_by_qtype", "Het", "tab:funnel_heterogeneity_by_qtype",
                 r"AI$\times$Post coefficient within each question type, by funnel stage (PPML, tag and week "
                 r"fixed effects). Figure~\ref{fig:funnel_heterogeneity_by_qtype} plots the coefficients with "
                 r"tag-clustered 95\% confidence intervals.",
                 r"Question type & Questions & Answered & Accepted & Acc., not closed & Acc., score $\geq0$ & Funnel-qualified",
                 het, ["questions", "answered", "accepted", "acc_nonclosed", "acc_nonneg", "funnel"],
                 ["coef"] * 6, "question_type")
    res = out["resolvability"].copy()
    table_simple(R, "table_resolvability_vs_quality", "Res", "tab:resolvability_vs_quality",
                 r"Triple-difference estimates on resolvability and quality shares at the cell level "
                 r"(weighted by the cell's question--tag pairs).",
                 r"Outcome & $\hat\beta$ & SE & $p$ & 95\% CI & N cells", res,
                 ["beta", "se", "p", "ci", "n"], ["scoef", "coef", "pv", "ci", "int"], "label")
    pg = out["prepost_groups"]
    t = T(R, "PP")
    L = [r"\begin{table}[ht]", r"\centering",
         r"\caption{Pre/post means by tag-index group (median split of the \RNTags{} tags on the pre-ChatGPT tag "
         r"index), weighted by question--tag pairs. $\Delta=\text{Post}-\text{Pre}$; "
         r"DiD $=\Delta_{\text{High}}-\Delta_{\text{Low}}$.}",
         r"\label{tab:prepost_resolvability_by_ai_group}", r"\footnotesize",
         r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Variable & Low pre & Low post & Low $\Delta$ & High pre & High post & High $\Delta$ & DiD \\",
         r"\midrule"]
    for var in pg.columns:
        lo_pre, lo_post = pg.loc[("LowAI", "Pre"), var], pg.loc[("LowAI", "Post"), var]
        hi_pre, hi_post = pg.loc[("HighAI", "Pre"), var], pg.loc[("HighAI", "Post"), var]
        vals = [lo_pre, lo_post, lo_post - lo_pre, hi_pre, hi_post, hi_post - hi_pre,
                (hi_post - hi_pre) - (lo_post - lo_pre)]
        lv, dv = ("int", "sint") if var.startswith("Body") else ("num3", "snum3")
        kinds = [lv, lv, dv, lv, lv, dv, dv]
        cols = ["LoPre", "LoPost", "LoD", "HiPre", "HiPost", "HiD", "DiD"]
        k = letters(var)[:20]
        L.append(f"{var} & " + " & ".join(t(k + c, v, kd) for c, v, kd in zip(cols, vals, kinds)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    write("table_prepost_resolvability_by_ai_group", L)
    ec = out["embed_corr"].copy()
    ec["label"] = ["Tag index"] + [f"Variant: {v} index" for v in AI_VARIANTS.values()]
    table_simple(R, "table_embedding_validation", "EmbV", "tab:embedding_validation",
                 r"Tag-level correlation between the embedding-based answerability score and the "
                 r"tag-index variants.", r"Index & Pearson $r$ & $p$ & Spearman $\rho$", ec,
                 ["pearson", "pearson_p", "spearman"], ["num3", "pv", "num3"], "label")
    table_simple(R, "table_embedding_partial_regression", "Part", "tab:embedding_partial_regression",
                 r"Tag-level regressions of the embedding-based answerability score (unstandardized) on the tag "
                 r"index with progressively added controls (HC1 standard errors).",
                 r"Specification & Coefficient & SE & $p$ & $R^2$ & Tags", out["partial"],
                 ["beta", "se", "p", "r2", "n"], ["num3", "num3", "pv", "num3", "int"], "spec")
    r_, p_, rho_, auc_, n_ = out["revealed_tag"]
    rv = pd.DataFrame([{"label": "Tag-level Pearson $r$ (index, mean revealed score)", "value": r_},
                       {"label": r"Tag-level Spearman $\rho$", "value": rho_},
                       {"label": "Question-level AUC (top tercile of revealed score)", "value": auc_}])
    table_simple(R, "table_revealed_answerability", "Rev", "tab:revealed_answerability",
                 r"Tag index and revealed answerability on a sample of pre-ChatGPT questions (one "
                 r"record per question). The revealed score shares inputs with the index (answering and "
                 r"acceptance), so the comparison describes the index; it is not an independent validation.",
                 r"Statistic & Value", rv,
                 ["value"], ["num3"], "label")
    table_simple(R, "table_auc_stratum", "AUC", "tab:auc_stratum",
                 r"Question-level discrimination of the tag index for high revealed "
                 r"answerability, overall and by quartile of the index.", r"Stratum & AUC & Questions",
                 out["auc_strata"], ["auc", "n"], ["num3", "int"], "stratum")


def tt(text: str) -> str:
    """Literal code text in a LaTeX table: backslashes and braces escaped, with
    break points after the alternatives of a regular expression."""
    out = text.replace("\\", "\x00").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
    out = out.replace("\x00", r"\textbackslash{}").replace("|", r"|\allowbreak{}")
    return r"\texttt{" + out + "}"


def table_classifier_rule(R: Registry) -> None:
    """The classification rule, written from the patterns and the threshold the pipeline applies."""
    from src.data.prepare_stackoverflow_question_type_raw import SHORT_BODY_MAX
    R.add("ClassBodyMax", SHORT_BODY_MAX, "int")
    zero = ", ".join(f"U+{ord(c):04X}" for c in ZERO_WIDTH_PAT.pattern.strip("[]"))
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Question-type classification rule, written from the code of the pipeline. A question takes the "
         r"first type, in this order, whose rule it meets. Regular expressions are matched on the title, ignoring "
         rf"case, after removing zero-width characters ({zero}). A question has code when its body contains a "
         r"\texttt{<code>} element; body length is in characters, HTML included. Sub is the substitutability label "
         r"of eq.~(\ref{eq:ddd}).}",
         r"\label{tab:classifier_rule}", r"\footnotesize", r"\begin{tabular}{rlcp{7.9cm}}", r"\toprule",
         r"Order & Question type & Sub & Rule \\", r"\midrule"]
    k = 0
    for name, pat in RULES:
        k += 1
        L.append(f"{k} & {tt(name)} & {0 if name in NON_SUBSTITUTABLE else 1} & {tt(pat.pattern)} \\\\")
    L += [rf"{k + 1} & {tt('short_code')} & 1 & Has code, body length $\leq$ \RClassBodyMax{{}} \\",
          rf"{k + 2} & {tt('long_code')} & 1 & Has code, body length $>$ \RClassBodyMax{{}} \\",
          rf"{k + 3} & {tt('other_conceptual')} & 1 & Every other question \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("table_classifier_rule", L)


def table_keyword_sensitivity(R: Registry, out: dict) -> None:
    t = T(R, "Loko")
    L = [r"{\footnotesize", r"\begin{longtable}{llrrrrr}",
         r"\caption{Leave-one-keyword-out on the title keywords of the two control types. Each row re-estimates "
         r"eq.~(\ref{eq:ddd}) with one keyword removed from the rule of Table~\ref{tab:classifier_rule}, so the "
         r"question--tag pairs whose only control keyword is that one join the substitutable types; VES is "
         r"version\_environment\_specific and AA advanced\_architecture; PPML, $p$-values "
         r"from tag-clustered standard errors (CR1).}\label{tab:keyword_sensitivity} \\", r"\toprule",
         r"Keyword left out & Type & Pairs moved & $\hat\beta$ volume & $p$ & $\hat\beta$ funnel-qualified & $p$ \\",
         r"\midrule", r"\endfirsthead", r"\toprule",
         r"Keyword left out & Type & Pairs moved & $\hat\beta$ volume & $p$ & $\hat\beta$ funnel-qualified & $p$ \\",
         r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for _, r in out["keyword_sensitivity"].iterrows():
        k = letters(r["keyword"])
        L.append(f"{r['keyword']} & {r['list']} & {t(k + 'Pairs', r['pairs'], 'int')} & {t(k + 'QB', r['q_beta'], 'coef')} & "
                 f"{t(k + 'QP', r['q_p'], 'pv')} & {t(k + 'FB', r['f_beta'], 'coef')} & {t(k + 'FP', r['f_p'], 'pv')} \\\\")
    L += [r"\end{longtable}", r"}"]
    write("table_keyword_sensitivity", L)


def table_minilm_panel() -> None:
    write("table_minilm_panel", [
        r"\begin{table}[!htbp]", r"\centering",
        r"\caption{Eq.~(\ref{eq:ddd}) with the substitutable label of the title-keyword classifier (baseline) and of "
        r"the \texttt{all-MiniLM-L6-v2} nearest-centroid classifier, with the exemplar questions of "
        r"\S\ref{sec:validation}, applied to the titles of all \RMiniLMNTitles{} unique questions of the main panel. "
        r"Inference as in the main text: CR1, score bootstrap and CR3.}",
        r"\label{tab:minilm_panel}", r"\small", r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Classifier and outcome & $\hat\beta_{\text{DDD}}$ & 95\% CI (CR1) & $p$ CR1 & $p$ score bootstrap & SE CR3 & $p$ CR3 \\",
        r"\midrule",
        r"Keywords, question volume & \RBaseB & [\RBaseLo, \RBaseHi] & \RBasePv & \RKSQPv & \RBaseCRThreeSE & \RBaseCRThreePv \\",
        r"Keywords, funnel-qualified posts & \RFunQualB & [\RFunQualLo, \RFunQualHi] & \RFunQualPv & \RKSQualPv & \RQualCRThreeSE & \RQualCRThreePv \\",
        r"MiniLM, question volume & \RMiniLMQB & [\RMiniLMQLo, \RMiniLMQHi] & \RMiniLMQPv & \RKSMiniLMQPv & \RMiniLMQCRThreeSE & \RMiniLMQCRThreePv \\",
        r"MiniLM, funnel-qualified posts & \RMiniLMQualB & [\RMiniLMQualLo, \RMiniLMQualHi] & \RMiniLMQualPv & \RKSMiniLMQualPv & \RMiniLMQualCRThreeSE & \RMiniLMQualCRThreePv \\",
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figures(D: Data, R: Registry, out: dict) -> None:
    w = out["weekly"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(w.index, w[1], color="#b2182b", lw=1.5, label="Substitutable")
    ax.plot(w.index, w[0], color="#2166ac", lw=1.5, label="Non-substitutable")
    ax.set_yscale("log")
    ax.axvline(CHATGPT, color="black", ls=":", lw=1)
    ax.set_ylabel("Question-tag pairs per week (log)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIGURES / "macro_substitutable_vs_not_log.png", dpi=160)
    plt.close(fig)

    es = out["es"]
    c = es[es["complete"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(c["bin"], c["beta"], yerr=[c["beta"] - c["ci_low"], c["ci_high"] - c["beta"]],
                fmt="o", color="black", ecolor="grey", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="#b2182b", ls=":", lw=1)
    ax.set_xlabel("Event time (13-week bins; bin $-1$ = reference)")
    ax.set_ylabel(r"Coefficient on AI$\times$Sub$\times$bin")
    fig.tight_layout()
    fig.savefig(FIGURES / "event_study_ddd_question_type_quarterly.png", dpi=160)
    plt.close(fig)

    c = out["es_qual"][out["es_qual"]["complete"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(c["bin"], c["beta"], yerr=[c["beta"] - c["ci_low"], c["ci_high"] - c["beta"]],
                fmt="o", color="black", ecolor="grey", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="#b2182b", ls=":", lw=1)
    ax.set_xlabel("Event time (13-week bins; bin $-1$ = reference)")
    ax.set_ylabel(r"Coefficient on AI$\times$Sub$\times$bin (funnel-qualified)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_event_study_funnel.png", dpi=160)
    plt.close(fig)

    by, theta = out["vtd"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(by["rank_act"], by["vtd"], s=18, c=np.where(by["flag"], "#b2182b", "grey"), alpha=0.75)
    ax.axhline(theta, ls="--", c="black", lw=1, label="Ratio for all plotted tags pooled")
    ax.set_xlabel("Tag rank by implied activity shortfall (1 = largest)")
    ax.set_ylabel("Implied upvoted-post shortfall / implied activity shortfall")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_vtd_monitor.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    sc = out["spec_curve"].sort_values("beta").reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 2]})
    axes[0].errorbar(sc.index, sc["beta"], yerr=[sc["beta"] - sc["ci_low"], sc["ci_high"] - sc["beta"]], fmt="o",
                     color="black", ecolor="grey", ms=4)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel(r"$\hat\beta_{\mathrm{DDD}}$")
    facs = [("sample", v) for v in ["full", "top50", "top75", "iqr_band"]] + \
           [("outcome", v) for v in ["questions", "unique_users"]] + [("control", v) for v in ["baseline", "excl_AA"]]
    for i, (f, v) in enumerate(facs):
        on = sc.index[sc[f] == v]
        axes[1].scatter(on, [i] * len(on), marker="s", color="black", s=14)
    axes[1].set_yticks(range(len(facs)))
    axes[1].set_yticklabels([f"{f}: {v}" for f, v in facs], fontsize=8)
    axes[1].set_xlabel("Specification (sorted by estimate)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_specification_curve.png", dpi=160)
    plt.close(fig)

    roll = out["rolling"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(roll["beta"], bins=15, color="grey", edgecolor="white")
    ax.axvline(out["base"].b, color="#b2182b", lw=1.5, label="Real cutoff")
    ax.set_xlabel("Placebo DDD coefficient")
    ax.set_ylabel("Number of placebo cutoffs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_rolling_placebo_distribution.png", dpi=160)
    plt.close(fig)

    tg = out["binscatter"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(tg[AI], tg["dlog"], s=16, color="black")
    sl, ic = np.polyfit(tg[AI], tg["dlog"], 1)
    xs = np.linspace(tg[AI].min(), tg[AI].max(), 50)
    ax.plot(xs, ic + sl * xs, color="#b2182b")
    ax.set_xlabel("Pre-ChatGPT tag index")
    ax.set_ylabel(r"$\Delta\log(1+Q)$, post minus pre (weekly means)")
    fig.tight_layout()
    fig.savefig(FIGURES / "binscatter_ai_effect.png", dpi=160)
    plt.close(fig)

    wk, post, _ = out["counterfactual"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wk["week_start"], wk["questions"], color="black", lw=1, label="Observed")
    ax.plot(post["week_start"], post["cf"], color="#2166ac", ls="--", lw=1.5, label="Log-linear extrapolation of pre-trend")
    ax.fill_between(post["week_start"], post["questions"], post["cf"], color="#b2182b", alpha=0.25)
    ax.axvline(CHATGPT, color="#b2182b", ls=":")
    ax.set_ylabel("Question-tag pairs per week")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "economic_magnitude_counterfactual.png", dpi=160)
    plt.close(fig)

    het = out["het"]
    stages = ["questions", "answered", "accepted", "acc_nonclosed", "acc_nonneg", "funnel"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, st in zip(axes.ravel(), stages):
        h = het[het["outcome"] == st].reset_index(drop=True)
        ax.errorbar(h["beta"], h.index, xerr=[h["beta"] - h["ci_low"], h["ci_high"] - h["beta"]], fmt="none",
                    ecolor="grey")
        ax.scatter(h["beta"], h.index, c=np.where(h["substitutable"] == 1, "#b2182b", "#333333"), zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(h.index)
        ax.set_yticklabels(h["question_type"], fontsize=7)
        ax.set_title(h["label"].iloc[0], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_funnel_heterogeneity_by_qtype.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    res = out["resolvability"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(res["beta"], range(len(res)), xerr=[res["beta"] - res["ci_low"], res["ci_high"] - res["beta"]],
                fmt="o", color="black", ecolor="grey")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res["label"])
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_resolvability_coefficients.pdf", metadata={"CreationDate": None})
    plt.close(fig)
