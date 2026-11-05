"""Shared constants, estimators and the result registry."""
from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[2]
RAW_SO = ROOT / "data" / "raw" / "stackoverflow"
PROCESSED = ROOT / "data" / "processed"
ANALYSIS = ROOT / "data" / "analysis"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
ESTIMATES = RESULTS / "estimates"

CHATGPT = pd.Timestamp("2022-11-30")
COPILOT = pd.Timestamp("2022-06-21")
AI = "ai_answerability_structural"
AI_VARIANTS = {
    "ai_answerability_zscore": "z-score composite",
    "ai_answerability_pca": "PCA composite",
    "ai_answerability_quantile": "quartile rank",
}
BIN_WEEKS = 13
SEED = 20260601


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------
@dataclass
class Est:
    b: float
    se: float
    p: float
    lo: float
    hi: float
    n: int

    def row(self, **extra) -> dict:
        return {"beta": self.b, "se": self.se, "p": self.p, "ci_low": self.lo,
                "ci_high": self.hi, "n": self.n, **extra}


def design(df: pd.DataFrame, ai: str = AI, cut: pd.Timestamp = CHATGPT,
           sub: str = "sub") -> pd.DataFrame:
    """Adds the triple-difference regressors of eq. (1)."""
    d = df.copy()
    d["ai"] = d[ai].astype(float)
    d["post"] = (d["week_start"] >= cut).astype(int)
    d["s"] = d[sub].astype(float)
    d["ai_post"] = d["ai"] * d["post"]
    d["ai_sub"] = d["ai"] * d["s"]
    d["post_sub"] = d["post"] * d["s"]
    d["aps"] = d["ai"] * d["post"] * d["s"]
    return d


def est_from(fit, name: str) -> Est:
    """A coefficient of a fit clustered by tag, with p-value and 95% CI from t(G-1),
    G the clusters in its sample (the smaller dimension under two-way clustering)."""
    b, se, df = float(fit.coef()[name]), float(fit.se()[name]), int(min(fit._G)) - 1
    q = tdist.ppf(0.975, df)
    return Est(b, se, float(2 * tdist.sf(abs(b / se), df)), b - q * se, b + q * se, int(fit._N))


def ddd(df: pd.DataFrame, y: str = "questions", ai: str = AI,
        cut: pd.Timestamp = CHATGPT, vcov=None, model: str = "ppml",
        weights: str | None = None, sub: str = "sub", extra: str = "",
        fe: str = "tag_qtype + week_id", return_fit: bool = False):
    """Eq. (1): AI x Post x Sub with the lower-order interactions, tag-by-type
    and week fixed effects, CR1 by tag. model="ppml" (count outcomes) is
    Poisson pseudo-maximum likelihood on Y; model="log" is OLS on log(1+Y),
    reported as robustness; model="ols" is OLS on Y, for shares."""
    d = design(df, ai, cut, sub)
    rhs = f"ai_post + ai_sub + post_sub + aps{extra} | {fe}"
    if model == "ppml":
        fit = pf.fepois(f"{y} ~ {rhs}", data=d, vcov=vcov or {"CRV1": "tag"})
    else:
        d["y_"] = np.log1p(d[y]) if model == "log" else d[y].astype(float)
        kw = {"weights": weights, "weights_type": "aweights"} if weights is not None else {}
        fit = pf.feols(f"y_ ~ {rhs}", data=d, vcov=vcov or {"CRV1": "tag"}, **kw)
    e = est_from(fit, "aps")
    if return_fit:
        return e, fit
    del fit, d
    gc.collect()                       # model fits hold reference cycles; free them now
    return e


def event_bins(week_start: pd.Series) -> pd.Series:
    """13-week bins anchored at the ChatGPT release; bin -1 is the 13 weeks
    immediately before the release week."""
    return ((week_start - CHATGPT).dt.days // 7) // BIN_WEEKS


def bin_name(b: int) -> str:
    return f"bn{abs(b)}" if b < 0 else f"bp{b}"


# --------------------------------------------------------------------------
# results registry -> results/numbers.tex
# --------------------------------------------------------------------------
def round_half_away(x: float, nd: int) -> Decimal:
    q = Decimal(1).scaleb(-nd)
    r = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return r + 0 if r == 0 else r          # a value that rounds to zero prints without a sign


def fmt_int(x: float) -> str:
    r = int(round_half_away(x, 0))
    s = f"{abs(r):,}".replace(",", "{,}")
    return f"\\ensuremath{{-{s}}}" if r < 0 else s


class Registry:
    """Unrounded values keyed by macro name; each kind fixes the format."""

    def __init__(self):
        self.vals: dict[str, tuple[object, str]] = {}

    def add(self, name: str, value, kind: str) -> None:
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        if name in self.vals:
            raise ValueError(f"duplicate macro {name}")
        self.vals[name] = (value, kind)

    def est(self, name: str, e: Est, signed: bool = False) -> None:
        self.add(f"{name}B", e.b, "scoef" if signed else "coef")
        self.add(f"{name}SE", e.se, "coef")
        self.add(f"{name}P", e.p, "p")
        self.add(f"{name}Pv", e.p, "pv")
        self.add(f"{name}Lo", e.lo, "coef")
        self.add(f"{name}Hi", e.hi, "coef")
        self.add(f"{name}N", e.n, "int")

    def get(self, name: str):
        return self.vals[name][0]

    # formatting ---------------------------------------------------------
    @staticmethod
    def render(value, kind: str) -> str:
        if kind in ("coef", "scoef"):
            r = round_half_away(value, 3)
            s = f"{abs(r):.3f}"
            if r < 0:
                return f"\\ensuremath{{-{s}}}"
            return f"\\ensuremath{{+{s}}}" if kind == "scoef" else s
        if kind in ("coef2", "scoef2"):
            r = round_half_away(value, 2)
            s = f"{abs(r):.2f}"
            if r < 0:
                return f"\\ensuremath{{-{s}}}"
            return f"\\ensuremath{{+{s}}}" if kind == "scoef2" else s
        if kind == "coef4":
            r = round_half_away(value, 4)
            return f"\\ensuremath{{-{abs(r):.4f}}}" if r < 0 else f"{r:.4f}"
        if kind == "p":
            if value < 0.001:
                return "\\ensuremath{p<0.001}"
            return f"\\ensuremath{{p={round_half_away(value, 3):.3f}}}"
        if kind == "pv":
            if value < 0.001:
                return "\\ensuremath{<0.001}"
            return f"{round_half_away(value, 3):.3f}"
        if kind == "int":
            return fmt_int(value)
        if kind == "sint":
            r = round_half_away(value, 0)
            return f"\\ensuremath{{-{fmt_int(abs(r))}}}" if r < 0 else f"\\ensuremath{{+{fmt_int(r)}}}"
        if kind == "snum3":
            r = round_half_away(value, 3)
            return f"\\ensuremath{{-{abs(r):.3f}}}" if r < 0 else f"\\ensuremath{{+{r:.3f}}}"
        if kind == "k":  # rounded to the nearest thousand
            return fmt_int(float(round_half_away(value / 1000, 0)) * 1000)
        if kind == "h":  # rounded to the nearest hundred
            return fmt_int(float(round_half_away(value / 100, 0)) * 100)
        if kind in ("pct1", "spct1"):
            r = round_half_away(100 * value, 1)
            if r < 0:
                return f"\\ensuremath{{-{abs(r):.1f}}}"
            return f"\\ensuremath{{+{r:.1f}}}" if kind == "spct1" else f"{r:.1f}"
        if kind == "num5":
            r = round_half_away(value, 5)
            return f"\\ensuremath{{-{abs(r):.5f}}}" if r < 0 else f"{r:.5f}"
        if kind == "spct0":
            r = round_half_away(100 * value, 0)
            return f"\\ensuremath{{-{abs(r):.0f}}}" if r < 0 else f"\\ensuremath{{+{r:.0f}}}"
        if kind == "pct0":
            return f"{round_half_away(100 * value, 0):.0f}"
        if kind == "num1":
            return f"{round_half_away(value, 1):.1f}"
        if kind == "num2":
            r = round_half_away(value, 2)
            return f"\\ensuremath{{-{abs(r):.2f}}}" if r < 0 else f"{r:.2f}"
        if kind == "num3":
            r = round_half_away(value, 3)
            return f"\\ensuremath{{-{abs(r):.3f}}}" if r < 0 else f"{r:.3f}"
        if kind == "text":
            return str(value)
        raise ValueError(kind)

    def write(self, tex_path: Path, csv_path: Path) -> None:
        lines = ["% Generated by run_pipeline.py; do not edit by hand."]
        rows = []
        for name in sorted(self.vals):
            value, kind = self.vals[name]
            lines.append(f"\\newcommand{{\\R{name}}}{{{self.render(value, kind)}}}")
            rows.append({"macro": f"\\R{name}", "kind": kind, "value": value})
        tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)


def stars(p: float) -> str:
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def tex_coef(x: float, nd: int = 3) -> str:
    r = round_half_away(x, nd)
    return f"$-{abs(r):.{nd}f}$" if r < 0 else f"{r:.{nd}f}"


def tex_p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{round_half_away(p, 3):.3f}"


def tex_int(x: float) -> str:
    return fmt_int(x)


def answerability_reference(index: pd.DataFrame) -> dict:
    """Bottom-decile reference for magnitudes that need a baseline tag: the
    tag closest to the 10th percentile of the structural index, and the
    10th and 90th percentiles of the index across tags."""
    ai = index.set_index("tag")[AI]
    p10, p90 = float(np.percentile(ai, 10)), float(np.percentile(ai, 90))
    tag = (ai - p10).abs().idxmin()
    return {"tag": tag, "ai": float(ai[tag]), "p10": p10, "p90": p90}


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def mde80(se: float) -> float:
    """Minimum detectable effect at 80% power, two-sided 5% test."""
    return 2.80 * se


def isclose(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, abs_tol=tol)
