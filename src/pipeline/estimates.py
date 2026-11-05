"""Estimates on the main panel: baseline, funnel, value margin,
event study, placebos, matched design, robustness and magnitude."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import linregress, pearsonr

from .common import (AI, AI_VARIANTS, BIN_WEEKS, CHATGPT, COPILOT, SEED, Est,
                     Registry, answerability_reference, bin_name, ddd, design, est_from,
                     event_bins, holm, mde80)
from .data import SCORE_BINS, Data, score_ratio
from src.data.prepare_answerability_index import STRUCTURAL_COMPONENTS, _zavg

FUNNEL = [("questions", "Questions", "FunQ"), ("answered", "Answered", "FunAns"),
          ("accepted", "Accepted answer", "FunAcc"),
          ("acc_nonclosed", "Accepted, not closed", "FunAccNC"),
          ("acc_nonneg", r"Accepted, score $\geq 0$", "FunAccNN"),
          ("funnel", "Funnel-qualified post", "FunQual")]
BIN_NAMES = ["Zero", "One", "Two", "Three", "Four", "FivePlus"]
GE_NAMES = ["Zero", "One", "Two", "Three", "Four", "Five"]
QTYPES = [("short_code", "Short"), ("how_to", "HowTo"), ("long_code", "Long"),
          ("debugging_simple", "Debug"), ("other_conceptual", "Other"),
          ("version_environment_specific", "VES"), ("advanced_architecture", "AA")]
PLACEBO_CUTS = ["2020-05-30", "2020-11-30", "2021-05-30", "2021-11-30", "2022-05-30"]
PLACEBO_NAMES = ["A", "B", "C", "D", "E"]


def implied_shortfall(p: pd.DataFrame, y: str, beta: float, ref: float, weight: str | None = None) -> float:
    """Cell-by-cell counterfactual minus observed over post-period cells,
    the counterfactual giving every tag the response of the reference tag
    (index value ref)."""
    post = p[p["week_start"] >= CHATGPT]
    mult = np.exp(-beta * (post[AI] - ref) * post["sub"]) - 1.0
    base = post[y] if weight is None else post[weight]
    return float((base * mult).sum())


def jackknife(df: pd.DataFrame, y: str, e: Est, ai: str = AI) -> tuple[Est, pd.Series]:
    """Leave-one-tag-out PPML estimates and the CR3 standard error built from
    them, centred on the full-sample estimate."""
    from scipy.stats import t as tdist
    tags = sorted(df["tag"].unique())
    G = len(tags)
    bj = pd.Series([ddd(df[df["tag"] != t], y=y, ai=ai).b for t in tags], index=tags)
    se3 = float(np.sqrt((G - 1) / G * ((bj - e.b) ** 2).sum()))
    q = tdist.ppf(0.975, G - 1)
    return Est(e.b, se3, float(2 * tdist.sf(abs(e.b / se3), G - 1)), e.b - q * se3, e.b + q * se3, e.n), bj


def leave_out_macros(R: Registry, nm: str, bj: pd.Series, b: float) -> None:
    """Estimate without python, the tag whose omission moves the estimate
    most, and the range of the leave-one-tag-out estimates."""
    R.add(f"NoPython{nm}B", bj["python"], "coef")
    R.add(f"LooAll{nm}Min", bj.min(), "coef")
    R.add(f"LooAll{nm}Max", bj.max(), "coef")
    R.add(f"TopInfluence{nm}", (bj - b).abs().idxmax(), "text")


def moved_panel(m: pd.DataFrame, rows: pd.DataFrame, to: str | None = None) -> pd.DataFrame:
    """Cell panel m with the question-tag pairs in rows moved to question type to,
    or removed from the panel when to is None."""
    keys = ["tag", "week_start", "question_type"]
    g = rows.groupby(keys).agg(questions=("question_id", "size"), funnel=("funnel", "sum"))
    out = m.set_index(keys)
    pos = out.index.get_indexer(g.index)
    cnt = {y: out[y].to_numpy(dtype=float, copy=True) for y in ("questions", "funnel")}
    for y, v in cnt.items():
        np.subtract.at(v, pos, g[y].to_numpy(dtype=float))
    if to is not None:
        dest = pd.MultiIndex.from_arrays([g.index.get_level_values(0), g.index.get_level_values(1), [to] * len(g)])
        pos2 = out.index.get_indexer(dest)
        for y, v in cnt.items():
            np.add.at(v, pos2, g[y].to_numpy(dtype=float))
    return out.assign(**cnt).reset_index()


def stacked_diff(df: pd.DataFrame, y0: str, y1: str, attenuation: bool = False):
    """beta(y1) - beta(y0) from one PPML model that stacks the two counts, with
    every regressor and both fixed effects interacted with the outcome, so the
    two coefficients equal the separate fits and the difference has a CR1-by-tag
    standard error that accounts for their correlation. With attenuation=True it
    also returns 1 - beta(y1)/beta(y0) and its 95% CI, by the delta method on the
    covariance of the stacked model."""
    d = design(df)
    long = pd.concat([d.assign(cnt=d[y0], o1=0, o="a"), d.assign(cnt=d[y1], o1=1, o="b")],
                     ignore_index=True)
    for v in ("ai_post", "post_sub", "aps"):
        long[f"{v}_1"] = long[v] * long["o1"]
    long["tq_o"] = long["tag_qtype"] + "_" + long["o"]
    long["wk_o"] = long["week_id"] + "_" + long["o"]
    fit = pf.fepois("cnt ~ aps + aps_1 + ai_post + ai_post_1 + post_sub + post_sub_1 | tq_o + wk_o",
                    data=long, vcov={"CRV1": "tag"})
    e = est_from(fit, "aps_1")
    if attenuation:
        names = list(fit.coef().index)
        i = [names.index("aps"), names.index("aps_1")]
        a, d = fit.coef().iloc[i].to_numpy()       # beta(y0) and beta(y1) - beta(y0)
        grad = np.array([d / a ** 2, -1 / a])
        se = float(np.sqrt(grad @ np.asarray(fit._vcov)[np.ix_(i, i)] @ grad))
        q = (e.hi - e.b) / e.se                    # critical value of the model's confidence intervals
        att = (float(-d / a), float(-d / a - q * se), float(-d / a + q * se))
    del fit, long
    gc.collect()
    return (e, att) if attenuation else e


def stacked_equality(df: pd.DataFrame, ys: list[str]) -> tuple[float, float, list[float]]:
    """Wald test that beta_DDD is the same for every outcome in ys, from one
    PPML model that stacks the counts with every regressor and both fixed
    effects interacted with the outcome (CR1 by tag). Returns the statistic,
    its p-value and the stacked coefficients, which equal the separate fits."""
    d = design(df)
    parts = []
    for j, y in enumerate(ys):
        p = d[["tag", "tag_qtype", "week_id"]].copy()
        p["cnt"] = d[y].to_numpy()
        for v in ("aps", "ai_post", "post_sub"):
            for k in range(len(ys)):
                p[f"{v}_{k}"] = d[v].to_numpy() * (k == j)
        p["tq_o"] = p["tag_qtype"] + f"_{j}"
        p["wk_o"] = p["week_id"] + f"_{j}"
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    rhs = " + ".join(f"{v}_{k}" for k in range(len(ys)) for v in ("aps", "ai_post", "post_sub"))
    fit = pf.fepois(f"cnt ~ {rhs} | tq_o + wk_o", data=long, vcov={"CRV1": "tag"})
    names = list(fit.coef().index)
    Rm = np.zeros((len(ys) - 1, len(names)))
    for k in range(1, len(ys)):
        Rm[k - 1, names.index("aps_0")], Rm[k - 1, names.index(f"aps_{k}")] = 1.0, -1.0
    wt = fit.wald_test(R=Rm)
    betas = [float(fit.coef()[f"aps_{k}"]) for k in range(len(ys))]
    del fit, long, parts
    gc.collect()
    return float(wt["statistic"]), float(wt["pvalue"]), betas


def score_bootstrap(df: pd.DataFrame, y: str, reps: int = 9999, seed: int = SEED, ai: str = AI) -> float:
    """Kline and Santos (2012) wild score bootstrap of H0: beta_DDD = 0 in the
    PPML model of eq. (1). The score of aps is evaluated at the restricted fit,
    partialled out of the nuisance parameters with the Poisson weights, summed
    by tag and perturbed with Rademacher signs; returns the two-sided p-value."""
    d = design(df, ai)
    # tag-by-type groups with no positive outcome have no score and are dropped by PPML
    d = d[d.groupby("tag_qtype")[y].transform("sum") > 0].reset_index(drop=True)
    r = pf.fepois(f"{y} ~ ai_post + ai_sub + post_sub | tag_qtype + week_id", data=d)
    mu = np.asarray(r.predict(type="response"), dtype=float)
    if len(mu) != len(d):
        raise RuntimeError("the restricted fit dropped observations; the score sample would not align")
    d["mu"] = mu
    xt = np.asarray(pf.feols("aps ~ ai_post + ai_sub + post_sub | tag_qtype + week_id",
                             data=d, weights="mu").resid(), dtype=float)
    if len(xt) != len(d):
        raise RuntimeError("the partialling-out regression dropped observations")
    s = pd.Series(xt * (d[y].to_numpy(dtype=float) - mu)).groupby(d["tag"].to_numpy()).sum().to_numpy()
    t0 = s.sum() / np.sqrt((s ** 2).sum())
    w = np.random.default_rng(seed).choice([-1.0, 1.0], size=(reps, len(s)))
    tb = (w @ s) / np.sqrt((s ** 2).sum())
    del r, d
    gc.collect()
    return float(np.mean(np.abs(tb) >= abs(t0)))


# --------------------------------------------------------------------------
def descriptives(D: Data, R: Registry, out: dict) -> None:
    m, q = D.main, D.q[D.q["week_start"].isin(D.weeks_main)]
    pre_w = D.weeks_main[D.weeks_main < CHATGPT]
    post_w = D.weeks_main[D.weeks_main >= CHATGPT]
    R.add("NTags", m["tag"].nunique(), "int")
    R.add("ClusterDf", m["tag"].nunique() - 1, "int")
    R.add("NTypes", m["question_type"].nunique(), "int")
    R.add("NWeeksMain", len(D.weeks_main), "int")
    R.add("NWeeksPre", len(pre_w), "int")
    R.add("NWeeksPost", len(post_w), "int")
    R.add("WeekFirst", D.weeks_main.min().strftime("%Y-%m-%d"), "text")
    R.add("MainLastDay", (D.weeks_main.max() + pd.Timedelta(days=6)).strftime("%Y-%m-%d"), "text")
    R.add("WeekLastStart", D.weeks_main.max().strftime("%Y-%m-%d"), "text")
    R.add("PostFirstWeek", post_w.min().strftime("%Y-%m-%d"), "text")
    R.add("PreLastDay", (pre_w.max() + pd.Timedelta(days=6)).strftime("%Y-%m-%d"), "text")
    R.add("NCellsMain", len(m), "int")
    R.add("CellsEmptyMain", (m["questions"] == 0).mean(), "pct1")
    R.add("NPairsMain", len(q), "int")
    R.add("NPairsMainM", len(q) / 1e6, "num1")
    R.add("NUniqueMain", q["question_id"].nunique(), "int")
    R.add("NUniqueMainM", q["question_id"].nunique() / 1e6, "num1")
    pre, post = q[q["week_start"] < CHATGPT], q[q["week_start"] >= CHATGPT]
    R.add("NPairsPost", len(post), "int")
    R.add("NUniquePost", post["question_id"].nunique(), "int")
    R.add("NPairsPre", len(pre), "int")
    R.add("NUniquePre", pre["question_id"].nunique(), "int")
    for wk, d in zip(["Start", "End"], [D.dropped_weeks.iloc[0], D.dropped_weeks.iloc[-1]]):
        R.add(f"DropWeek{wk}", d["week_start"].strftime("%Y-%m-%d"), "text")
        R.add(f"DropWeek{wk}Days", d["days"], "int")
    R.add("IndexWinStart", D.index_window[0].strftime("%Y-%m-%d"), "text")
    R.add("IndexWinEnd", D.index_window[1].strftime("%Y-%m-%d"), "text")
    # calendar-year volume (question-tag pairs and unique questions)
    yr = q["creation_date"].dt.year
    for y_, nm in [(2020, "TwentyTwenty"), (2024, "TwentyTwentyFour")]:
        R.add(f"Pairs{nm}M", (yr == y_).sum() / 1e6, "num1")
        R.add(f"Unique{nm}M", q.loc[yr == y_, "question_id"].nunique() / 1e6, "num1")
    # weekly means (question-tag pairs)
    wk = m.groupby("week_start")["questions"].sum()
    wpre, wpost = wk[wk.index < CHATGPT].mean(), wk[wk.index >= CHATGPT].mean()
    R.add("WeeklyPre", wpre, "h")
    R.add("WeeklyPost", wpost, "h")
    R.add("WeeklyChg", wpost / wpre - 1, "spct1")
    for s, nm in [(1, "Sub"), (0, "NonSub")]:
        w = m[m["sub"] == s].groupby("week_start")["questions"].sum()
        R.add(f"WeeklyChg{nm}", w[w.index >= CHATGPT].mean() / w[w.index < CHATGPT].mean() - 1, "spct1")
    tq = D.index.set_index("tag")[AI].rank(method="first")
    quart = pd.qcut(tq, 4, labels=[1, 2, 3, 4])
    mq = m.assign(qrt=m["tag"].map(quart))
    for k, nm in [(1, "One"), (2, "Two"), (3, "Three"), (4, "Four")]:
        w = mq[mq["qrt"] == k].groupby("week_start")["questions"].sum()
        R.add(f"QuartChg{nm}", w[w.index >= CHATGPT].mean() / w[w.index < CHATGPT].mean() - 1, "spct1")
    # tag-level change in log weekly volume vs the index
    tw = m.groupby(["tag", "week_start"])["questions"].sum().reset_index()
    tw["post"] = tw["week_start"] >= CHATGPT
    tm = tw.groupby(["tag", "post"])["questions"].mean().unstack()
    dl = (np.log1p(tm[True]) - np.log1p(tm[False])).rename("dlog")
    tg = D.index.set_index("tag")[AI].to_frame().join(dl)
    r, p = pearsonr(tg["dlog"], tg[AI])
    R.add("TagCorrR", r, "num2")
    R.add("TagCorrP", p, "p")
    out["binscatter"] = tg
    # funnel survival and score shares (question-tag pairs)
    R.add("FunnelSurvive", m["funnel"].sum() / m["questions"].sum(), "pct1")
    R.add("ShareGeOne", m["ge1"].sum() / m["funnel"].sum(), "pct0")
    R.add("ShareGeFive", m["ge5"].sum() / m["funnel"].sum(), "pct0")
    out["weekly"] = m.groupby(["week_start", "sub"])["questions"].sum().unstack()


def source_validation(D: Data, R: Registry, out: dict) -> None:
    """API pull versus SEDE extract for the same day (2023-05-02)."""
    for k in ["ValAPIRows", "ValSEDERows", "ValUnmatched", "ValTitleEscaped", "ValTitleOther",
              "ValBodyDiff", "ValOtherDiff", "ValTypeDiff"]:
        R.add(k, int(D.counts[k]), "int")
    R.add("NTagsTwentyFive", D.q.loc[D.q["creation_date"] >= "2025-01-01", "tag"].nunique(), "int")


# --------------------------------------------------------------------------
def baseline(D: Data, R: Registry, out: dict) -> None:
    m = D.main
    e = ddd(m)
    R.est("Base", e)
    R.add("BaseT", e.b / e.se, "num2")
    out["base"] = e
    cr3, bj = jackknife(m, "questions", e)
    R.est("BaseCRThree", cr3)
    out["jack_q"] = bj
    leave_out_macros(R, "Q", bj, e.b)
    vol = m.groupby("tag")["questions"].sum()
    R.add("LargestTag", vol.idxmax(), "text")
    R.add("LargestTagShare", vol.max() / vol.sum(), "pct1")
    noaa = ddd(m[m["question_type"] != "advanced_architecture"])
    R.est("NoAA", noaa)
    out["noaa"] = noaa
    recode = m.assign(sub=(m["question_type"] != "version_environment_specific").astype(int))
    rec = ddd(recode)
    R.est("AARecode", rec)
    # each category as the sole control
    rows = [{"control": "baseline (VES and advanced_architecture)", **e.row()}]
    for qt, nm in QTYPES:
        e_s = ddd(m.assign(sub=(m["question_type"] != qt).astype(int)))
        R.est(f"Sole{nm}", e_s, signed=True)
        rows.append({"control": qt, **e_s.row()})
    cf = pd.DataFrame(rows)
    out["control_fragility"] = cf
    R.add("SoleCtrlMin", cf["beta"].iloc[1:].min(), "coef")
    R.add("SoleCtrlMax", cf["beta"].iloc[1:].max(), "scoef")
    out["cr3"] = cr3
    # alternative outcomes and inference
    out["users"] = ddd(m, y="unique_users")
    out["frac"] = ddd(m, y="q_frac")
    out["twoway"] = ddd(m, vcov={"CRV1": "tag+week_id"})
    out["log_base"] = ddd(m, model="log")
    for nm, k in [("Users", "users"), ("Frac", "frac"), ("TwoWay", "twoway"), ("LogBase", "log_base")]:
        R.est(nm, out[k])
    out["ks_q"] = score_bootstrap(m, "questions")
    R.add("KSQP", out["ks_q"], "p")
    R.add("KSQPv", out["ks_q"], "pv")
    R.add("KSReps", 9999, "int")
    # index variants (robustness table only)
    out["variants"] = {k: ddd(m, ai=k) for k in AI_VARIANTS}
    for k, nm in zip(AI_VARIANTS, ["VarZ", "VarPCA", "VarQuart"]):
        R.est(nm, out["variants"][k])
    # sample restrictions by pre-period volume
    vol = m[m["week_start"] < CHATGPT].groupby("tag")["questions"].sum().sort_values(ascending=False)
    out["vol_rank"] = vol
    for n, nm in [(50, "TopFifty"), (75, "TopSeventyFive")]:
        out[f"top{n}"] = ddd(m[m["tag"].isin(vol.index[:n])])
        R.est(nm, out[f"top{n}"])
    loo = {}
    for n in [1, 5, 10, 20]:
        loo[n] = ddd(m[~m["tag"].isin(vol.index[:n])])
    jack = {t: ddd(m[m["tag"] != t]) for t in vol.index[:10]}
    out["loo"], out["jack"] = loo, jack
    R.add("TopTag", vol.index[0], "text")
    for n, nm in [(1, "One"), (5, "Five"), (10, "Ten"), (20, "Twenty")]:
        R.est(f"Drop{nm}", loo[n])
    R.add("LooMin", min(x.b for x in loo.values()), "coef")
    R.add("LooMax", max(x.b for x in loo.values()), "coef")
    R.add("JackMin", min(x.b for x in jack.values()), "coef")
    R.add("JackMax", max(x.b for x in jack.values()), "coef")


# --------------------------------------------------------------------------
def funnel(D: Data, R: Registry, out: dict) -> None:
    m = D.main
    qm = D.q[D.q["week_start"].isin(D.weeks_main)]
    qpost = qm[qm["week_start"] >= CHATGPT]
    ref = answerability_reference(D.index)
    out["reference"] = ref
    R.add("RefTag", ref["tag"], "text")
    R.add("AIPTen", ref["p10"], "num2")
    R.add("AIPNinety", ref["p90"], "num2")
    rows = []
    for col, label, nm in FUNNEL:
        e, fit = ddd(m, y=col, return_fit=True)
        g3 = est_from(fit, "post_sub")          # within-tag gap at the zero of the tag index
        del fit
        gc.collect()
        sf = implied_shortfall(m, col, e.b, ref["ai"])
        rows.append({"outcome": col, "label": label, **e.row(), "shortfall": sf,
                     "g3": g3.b, "g3_se": g3.se, "g3_p": g3.p})
        if nm != "FunQ":
            R.est(nm, e)
        if nm in ("FunQ", "FunQual"):
            R.est("GammaThree" + ("Q" if nm == "FunQ" else "Qual"), g3, signed=True)
    fun = pd.DataFrame(rows)
    out["funnel"] = fun
    # normalization-free magnitude: proportional effect of moving from the 10th to the 90th percentile
    delta = ref["p90"] - ref["p10"]
    for nm, r in [("Q", fun.loc[0]), ("Qual", fun.loc[5])]:
        R.add(f"Prop{nm}", np.expm1(r["beta"] * delta), "spct1")
        R.add(f"Prop{nm}Lo", np.expm1(r["ci_low"] * delta), "spct1")
        R.add(f"Prop{nm}Hi", np.expm1(r["ci_high"] * delta), "spct1")
    # inference for the funnel-qualified coefficient: jackknife over tags
    qual = Est(*fun.loc[5, ["beta", "se", "p", "ci_low", "ci_high", "n"]].tolist())
    cr3f, bjf = jackknife(m, "funnel", qual)
    R.est("QualCRThree", cr3f)
    leave_out_macros(R, "Qual", bjf, qual.b)
    out["cr3_qual"], out["jack_qual"] = cr3f, bjf
    R.add("FunMin", fun["beta"].min(), "coef")
    R.add("FunMax", fun["beta"].max(), "coef")
    R.add("FunPMax", fun["p"].max(), "pv")
    # value-weighted outcomes
    vw = []
    for col, label, nm in [("funnel", "Funnel-qualified post (count)", "VWQual"),
                           ("ge1", r"Upvoted (score $\geq 1$)", "VWHigh"),
                           ("ge5", r"Score $\geq 5$", "VWVery"),
                           ("value_w", r"Value-weighted ($\sum\log(1+\max(s,0))$)", "VWValue")]:
        e = ddd(m, y=col)
        vw.append({"outcome": col, "label": label, **e.row(),
                   "shortfall": implied_shortfall(m, col, e.b, ref["ai"])})
        if nm != "VWQual":
            R.est(nm, e)
    out["value_weighted"] = pd.DataFrame(vw)
    high = out["value_weighted"].loc[1]
    R.add("VWHighMDE", mde80(high["se"]), "num3")
    # one effect at every stage: funnel-qualified minus questions in one stacked model
    eq, (att, att_lo, att_hi) = stacked_diff(m, "questions", "funnel", attenuation=True)
    R.est("FunEq", eq, signed=True)
    out["funnel_equality"] = eq
    R.add("FunAtten", att, "pct0")
    R.add("FunAttenLo", att_lo, "spct0")
    R.add("FunAttenHi", att_hi, "pct0")
    # one effect at every stage: joint Wald test across the six stages
    stat, pj, betas = stacked_equality(m, [c for c, _, _ in FUNNEL])
    R.add("FunJointChi", stat, "num2")
    R.add("FunJointP", pj, "p")
    R.add("FunJointPv", pj, "pv")
    R.add("FunJointDf", len(FUNNEL) - 1, "int")
    out["funnel_joint"] = pd.DataFrame({"outcome": [c for c, _, _ in FUNNEL], "stacked_beta": betas,
                                        "separate_beta": fun["beta"].to_numpy(), "wald": stat, "p": pj})
    # vote margin within funnel-qualified posts: score 0 versus score >= 1
    zero = ddd(m, y="s0")
    R.est("VoteZero", zero)
    vd = stacked_diff(m, "s0", "ge1")
    R.est("VoteDiff", vd, signed=True)
    out["vote_margin"] = {"zero": zero, "high": high, "diff": vd}
    # Kline-Santos score bootstrap for the volume, funnel-qualified and score >= 1 tests
    for nm, y in [("Qual", "funnel"), ("High", "ge1")]:
        p = score_bootstrap(m, y)
        R.add(f"KS{nm}P", p, "p")
        R.add(f"KS{nm}Pv", p, "pv")
        out[f"ks_{nm.lower()}"] = p
    # log(1+Y) robustness for the funnel and the vote margin
    rows = []
    for col, label in [*[(c, lab) for c, lab, _ in FUNNEL], ("s0", "Funnel-qualified, score 0"),
                       ("ge1", r"Funnel-qualified, score $\geq 1$")]:
        lg = ddd(m, y=col, model="log")
        pp = (fun.set_index("outcome").loc[col] if col in fun["outcome"].values
              else (zero.row() if col == "s0" else high))
        rows.append({"outcome": col, "label": label, "ppml_beta": pp["beta"], "ppml_p": pp["p"], **lg.row()})
    out["log_robust"] = pd.DataFrame(rows)

    # shortfalls in question-tag pairs and unique-question equivalents
    def short(e_b: float, flag: str | None):
        k = qpost if flag is None else qpost[qpost[flag] == 1]
        mult = np.exp(-e_b * (k[AI].to_numpy() - ref["ai"]) * k["substitutable_type"].to_numpy()) - 1.0
        return float(mult.sum()), float((mult / k["k_tags"].to_numpy()).sum()), len(k), k["question_id"].nunique()

    qpost = qpost.merge(D.index[["tag", AI]], on="tag", how="left")
    qpost["hv1"] = (qpost["s1"] + qpost["s2"] + qpost["s3"] + qpost["s4"] + qpost["s5p"]).astype(int)
    b_q, b_f = fun.loc[0, "beta"], fun.loc[5, "beta"]
    b_h = out["value_weighted"].loc[1, "beta"]
    for nm, b, flag in [("Q", b_q, None), ("Qual", b_f, "funnel"), ("HV", b_h, "hv1")]:
        pairs, uniq, n_pairs, n_uniq = short(b, flag)
        R.add(f"Short{nm}", pairs, "int")
        R.add(f"Short{nm}K", pairs, "k")
        R.add(f"Short{nm}H", pairs, "h")
        R.add(f"Short{nm}U", uniq, "int")
        R.add(f"Short{nm}UK", uniq, "k")
        R.add(f"Short{nm}UH", uniq, "h")
        R.add(f"Post{nm}Pairs", n_pairs, "int")
        R.add(f"Post{nm}Unique", n_uniq, "int")
        R.add(f"Short{nm}Pct", pairs / n_pairs, "pct1")
        R.add(f"Short{nm}UPct", uniq / n_uniq, "pct1")
        out[f"short_{nm}"] = (pairs, uniq)
    # each shortfall over the 95% CI of its coefficient: the image of the interval
    for nm, r, flag in [("Q", fun.loc[0], None), ("Qual", fun.loc[5], "funnel"), ("HV", high, "hv1")]:
        grid = np.linspace(r["ci_low"], r["ci_high"], 201)
        sv = np.array([short(b, flag)[:2] for b in grid])
        for j, unit in [(0, ""), (1, "U")]:
            R.add(f"Short{nm}{unit}Lo", sv[:, j].min(), "int")
            R.add(f"Short{nm}{unit}Hi", sv[:, j].max(), "int")
    sq, sf_, sh = out["short_Q"], out["short_Qual"], out["short_HV"]
    R.add("ShortHVShareAll", sh[0] / len(qpost), "pct1")
    R.add("ShortRatio", sq[0] / sf_[0], "num1")
    R.add("ShortRatioU", sq[1] / sf_[1], "num1")
    R.add("ShortQualShare", sf_[0] / sq[0], "pct0")
    weeks_post = int((D.weeks_main >= CHATGPT).sum())
    for nm, v in [("HVFlowYear", sh[0]), ("HVFlowYearU", sh[1]),
                  ("QualFlowYear", sf_[0]), ("QualFlowYearU", sf_[1])]:
        R.add(nm, v / weeks_post * 52, "h")
        R.add(nm.replace("Year", "FiveYears"), v / weeks_post * 52 * 5, "k")
    # funnel excluding advanced_architecture
    na = m[m["question_type"] != "advanced_architecture"]
    out["funnel_noaa"] = pd.DataFrame([{"label": label, **(e := ddd(na, y=col)).row(),
                                        "shortfall": implied_shortfall(na, col, e.b, ref["ai"])}
                                       for col, label, _ in FUNNEL])
    R.est("NoAAQual", Est(*out["funnel_noaa"].loc[5, ["beta", "se", "p", "ci_low", "ci_high", "n"]].tolist()))


# --------------------------------------------------------------------------
def score_bins(D: Data, R: Registry, out: dict) -> None:
    """The six mutually exclusive score bins of funnel-qualified posts, with
    Holm-adjusted p-values, as a description of the shape; the
    voting-normalised thresholds; and the log(1+Y) bins against the
    funnel-qualified count."""
    m = D.main
    rows = []
    for col, nm in zip(SCORE_BINS, BIN_NAMES):
        e = ddd(m, y=col)
        R.est(f"Bin{nm}", e)
        R.add(f"Bin{nm}MDE", mde80(e.se), "num2")
        rows.append({"bin": col, **e.row(), "mde80": mde80(e.se), "posts": int(m[col].sum())})
    bins = pd.DataFrame(rows)
    bins["p_holm"] = holm(bins["p"])
    for nm, ph in zip(BIN_NAMES, bins["p_holm"]):
        R.add(f"Bin{nm}PHolm", ph, "pv")
    out["bins"] = bins
    # bin coefficients averaged with weights equal to each bin's share of funnel-qualified posts
    w = bins["posts"].to_numpy(dtype=float)
    w /= w.sum()
    log_est = [ddd(m, y=c, model="log") for c in SCORE_BINS]
    log_bins = np.array([e.b for e in log_est])
    bin_labels = [None, "Funnel-qualified, score 1", "Funnel-qualified, score 2", "Funnel-qualified, score 3",
                  "Funnel-qualified, score 4", r"Funnel-qualified, score $\geq 5$"]
    extra = [{"outcome": c, "label": lab, "ppml_beta": bins.loc[i, "beta"], "ppml_p": bins.loc[i, "p"],
              **log_est[i].row()} for i, (c, lab) in enumerate(zip(SCORE_BINS, bin_labels)) if lab]
    out["log_robust"] = pd.concat([out["log_robust"], pd.DataFrame(extra)], ignore_index=True)
    R.add("PPBinsWAvg", float(w @ bins["beta"].to_numpy()), "coef")
    R.add("LogBinsWAvg", float(w @ log_bins), "coef")
    R.add("LogFunQualB", float(out["log_robust"].set_index("outcome").loc["funnel", "beta"]), "coef")
    # voting-normalised score thresholds
    vn = []
    for k in (1, 2, 5):
        e = ddd(m, y=f"vn{k}")
        vn.append({"k": k, **e.row(), "posts": int(m[f"vn{k}"].sum())})
    out["voting"] = pd.DataFrame(vn)
    clips = []
    for lab, suf in [("no clip", "_noclip"), ("[0.3, 5.0]", "_c35"), ("[0.5, 3.0]", ""), ("[0.7, 2.0]", "_c72")]:
        rec = {"clip": lab}
        for k in (1, 2, 5):
            e = ddd(m, y=f"vn{k}{suf}")
            rec[f"k{k}_beta"], rec[f"k{k}_p"] = e.b, e.p
        clips.append(rec)
    out["clip"] = pd.DataFrame(clips)
    # the text states that no clip range changes any estimate
    if float(out["clip"].drop(columns="clip").sub(out["clip"].drop(columns="clip").iloc[0]).abs().max().max()) > 1e-12:
        raise RuntimeError("voting-normalised estimates depend on the clip range")
    ratio = score_ratio(D.q[D.q["week_start"].isin(D.weeks_main)])
    # tags for which any clip range binds ([0.7, 2.0] is the narrowest)
    R.add("ClipNBind", int(((ratio < 0.7) | (ratio > 2.0)).sum()), "int")


# --------------------------------------------------------------------------
def event_study(df: pd.DataFrame, ai: str = AI, controls: str = " + ai_post + post_sub", y: str = "questions"):
    """Dynamic version of eq. (1), estimated by PPML: AI x Sub interacted with
    13-week bins (reference: bin -1), holding the DDD's lower-order terms fixed."""
    d = design(df, ai)
    d["bin"] = event_bins(d["week_start"])
    bins = sorted(int(b) for b in d["bin"].unique())
    terms = [bin_name(b) for b in bins if b != -1]
    for b in bins:
        if b != -1:
            d[bin_name(b)] = d["ai"] * d["s"] * (d["bin"] == b)
    fit = pf.fepois(f"{y} ~ {' + '.join(terms)}{controls} | tag_qtype + week_id",
                    data=d, vcov={"CRV1": "tag"})
    wk = d.groupby("bin")["week_start"].agg(["min", "max", "nunique"])
    rows = []
    for b in bins:
        nm = bin_name(b)
        r = {"bin": b, "first_week": wk.loc[b, "min"], "last_day": wk.loc[b, "max"] + pd.Timedelta(days=6),
             "n_weeks": int(wk.loc[b, "nunique"]), "complete": int(wk.loc[b, "nunique"]) == BIN_WEEKS}
        if b == -1:
            r.update(beta=0.0, se=0.0, p=np.nan, ci_low=0.0, ci_high=0.0)
        else:
            e = est_from(fit, nm)
            r.update(beta=e.b, se=e.se, p=e.p, ci_low=e.lo, ci_high=e.hi)
        rows.append(r)
    names = list(fit.coef().index)
    V = pd.DataFrame(np.asarray(fit._vcov), index=names, columns=names)
    return pd.DataFrame(rows), V, fit, d, terms


def dynamics(D: Data, R: Registry, out: dict) -> None:
    controls = " + ai_post + post_sub"
    es, V, fit, _, _ = event_study(D.main, controls=controls)
    out["es"], out["es_V"] = es, V
    comp = es[es["complete"]]
    post = comp[comp["bin"] >= 0]
    pre = comp[(comp["bin"] < -1)]
    for _, r in comp[comp["bin"] != -1].iterrows():
        nm = f"ES{'Pre' if r['bin'] < 0 else 'Post'}{abs(int(r['bin']))}"
        nm = nm.replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three") \
            .replace("4", "Four").replace("5", "Five").replace("6", "Six").replace("7", "Seven") \
            .replace("8", "Eight").replace("9", "Nine")
        R.est(nm, Est(r["beta"], r["se"], r["p"], r["ci_low"], r["ci_high"], int(fit._N)))
    R.add("ESNPost", len(post), "int")
    R.add("ESNPre", len(pre), "int")
    R.add("ESLastCompleteEnd", post["last_day"].max().strftime("%Y-%m-%d"), "text")
    R.add("ESFirstCompleteStart", pre["first_week"].min().strftime("%Y-%m-%d"), "text")
    R.add("ESPostMin", post["beta"].min(), "coef")
    R.add("ESPostMax", post["beta"].max(), "coef")
    R.add("ESPostPMax", post["p"].max(), "pv")
    ps = post.sort_values("bin")["p"].to_numpy()
    first_sig = next((i for i in range(len(ps)) if (ps[i:] < 0.05).all()), None)
    out["es_first_sig"] = None if first_sig is None else int(post.sort_values("bin")["bin"].iloc[first_sig])
    if first_sig is not None:
        R.add("ESFirstSigBin", out["es_first_sig"], "int")
    R.add("ESPreMean", pre["beta"].mean(), "coef")
    R.add("ESPostMean", post["beta"].mean(), "coef")
    out["es_post_monotone"] = bool(np.all(np.diff(post.sort_values("bin")["beta"].to_numpy()) < 0))
    pre_terms = [bin_name(int(b)) for b in pre["bin"]]
    wt = fit.wald_test(R=np.eye(len(fit.coef()))[[list(fit.coef().index).index(t) for t in pre_terms]])
    R.add("ESPreJointP", float(wt["pvalue"]), "p")
    R.add("ESPreJointChi", float(wt["statistic"]), "num2")
    theta = float(post["beta"].mean())
    R.add("Theta", theta, "coef")
    out["theta_bins"] = [bin_name(int(b)) for b in post["bin"]]
    out["honest_pre_bins"] = pre_terms


def rscript() -> str:
    import glob
    import os
    import shutil
    found = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if not found:
        cands = sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"))
        found = cands[-1] if cands else None
    if not found:
        raise RuntimeError("Rscript not found; set RSCRIPT")
    return found


def honestdid(D: Data, R: Registry, out: dict) -> None:
    """Sensitivity of theta (average of the complete post-period event-study
    bins) to violations of parallel trends, via src/pipeline/honestdid.R."""
    import subprocess
    from .common import ESTIMATES
    V, es = out["es_V"], out["es"]
    pre, post = out["honest_pre_bins"], out["theta_bins"]
    names = pre + post
    d = ESTIMATES / "honestdid"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"term": names, "beta": [es.set_index(es["bin"].map(bin_name)).loc[n, "beta"] for n in names]}
                 ).to_csv(d / "honestdid_betahat.csv", index=False)
    np.savetxt(d / "honestdid_sigma.csv", V.loc[names, names].to_numpy(), delimiter=",")
    pd.DataFrame([{"n_pre": len(pre), "n_post": len(post), "df": D.main["tag"].nunique() - 1}]
                 ).to_csv(d / "honestdid_meta.csv", index=False)
    subprocess.run([rscript(), str(Path(__file__).with_name("honestdid.R")), str(d)], check=True)
    hd = pd.read_csv(d / "honestdid_results.csv")
    orig = pd.read_csv(d / "honestdid_original.csv").iloc[0]
    out["honestdid"], out["honestdid_orig"] = hd, orig
    R.add("HDTheta", orig["theta"], "coef")
    R.add("HDThetaSE", orig["se"], "coef")
    R.add("HDOrigLo", orig["orig_lb"], "coef")
    R.add("HDOrigHi", orig["orig_ub"], "coef")
    R.add("HDOrigMid", (orig["orig_lb"] + orig["orig_ub"]) / 2, "coef")
    R.add("HDNPre", len(pre), "int")
    R.add("HDNPost", len(post), "int")
    R.add("HDVersion", orig["version"], "text")
    words = {0.0: "Zero", 0.25: "QuarterM", 0.5: "Half", 0.75: "ThreeQuarters", 1.0: "One",
             1.5: "OneHalf", 2.0: "Two"}
    for _, r in hd.iterrows():
        R.add(f"HDLo{words[r['Mbar']]}", r["lb"], "coef")
        R.add(f"HDHi{words[r['Mbar']]}", r["ub"], "scoef")
    cont = hd[(hd["lb"] <= 0) & (hd["ub"] >= 0)]
    bd = pd.read_csv(d / "honestdid_rm_breakdown.csv").iloc[0]
    out["hd_rm_breakdown"] = float(bd["breakdown"])        # 0: includes zero at Mbar = 0; NaN: beyond the grid
    if np.isfinite(out["hd_rm_breakdown"]):
        R.add("HDBreakdown", out["hd_rm_breakdown"], "num3")
    R.add("HDBreakdownStep", float(bd["step"]), "num3")
    R.add("HDThetaGridStep", float(bd["theta_step"]), "num5")
    # raw second differences of the pre-period path, with bin -1 as the reference, in the unit of M
    path = es.set_index(es["bin"].map(bin_name)).loc[pre, "beta"].to_numpy()
    raw = np.abs(np.diff(np.r_[path, 0.0], 2))
    R.add("PreRawDiffMin", raw.min(), "num3")
    R.add("PreRawDiffMax", raw.max(), "num3")
    last_ex = hd[hd["Mbar"] < cont["Mbar"].min()]["Mbar"].max() if len(cont) else hd["Mbar"].max()
    R.add("HDLastExcl", float(last_ex), "num2")
    le = hd[hd["Mbar"] == last_ex].iloc[0]
    R.add("HDLoLast", le["lb"], "coef")
    R.add("HDHiLast", le["ub"], "scoef")
    sd = pd.read_csv(d / "honestdid_sd_results.csv")
    out["honestdid_sd"] = sd
    z = sd[sd["M"] == 0].iloc[0]
    R.add("HDSDLoZero", z["lb"], "coef")
    R.add("HDSDHiZero", z["ub"], "coef")
    R.add("HDSDCenterZero", (z["lb"] + z["ub"]) / 2, "coef")      # FLCI is symmetric
    bd_sd = pd.read_csv(d / "honestdid_sd_breakdown.csv").iloc[0]
    out["hd_sd_breakdown"] = float(bd_sd["breakdown"])
    if np.isfinite(out["hd_sd_breakdown"]):
        R.add("HDSDBreakdown", out["hd_sd_breakdown"], "num5")
        R.add("HDSDBreakdownStep", float(bd_sd["step"]), "num5")


def honestdid_run(R: Registry, nm: str, es: pd.DataFrame, V: pd.DataFrame, pre_terms: list[str],
                  post_terms: list[str], dirname: str, df: int, lvec: np.ndarray | None = None) -> None:
    """Runs honestdid.R on an event study, for the equal-weighted average of the
    post-period bins or for the weights in lvec, and registers its macros."""
    import subprocess
    from .common import ESTIMATES
    names = pre_terms + post_terms
    d = ESTIMATES / dirname
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"term": names, "beta": [es.set_index(es["bin"].map(bin_name)).loc[n, "beta"] for n in names]}
                 ).to_csv(d / "honestdid_betahat.csv", index=False)
    np.savetxt(d / "honestdid_sigma.csv", V.loc[names, names].to_numpy(), delimiter=",")
    pd.DataFrame([{"n_pre": len(pre_terms), "n_post": len(post_terms), "df": df}]).to_csv(d / "honestdid_meta.csv", index=False)
    if lvec is not None:
        pd.DataFrame({"l": lvec}).to_csv(d / "honestdid_lvec.csv", index=False)
    subprocess.run([rscript(), str(Path(__file__).with_name("honestdid.R")), str(d)], check=True)
    orig = pd.read_csv(d / "honestdid_original.csv").iloc[0]
    R.add(f"{nm}Theta", orig["theta"], "coef")
    R.add(f"{nm}OrigLo", orig["orig_lb"], "coef")
    R.add(f"{nm}OrigHi", orig["orig_ub"], "coef")
    bd = float(pd.read_csv(d / "honestdid_rm_breakdown.csv")["breakdown"].iloc[0])
    if np.isfinite(bd):
        R.add(f"{nm}Breakdown", bd, "num3")
    sd = pd.read_csv(d / "honestdid_sd_results.csv")
    z = sd[sd["M"] == 0].iloc[0]
    R.add(f"{nm}SDLoZero", z["lb"], "coef")
    R.add(f"{nm}SDHiZero", z["ub"], "coef")
    R.add(f"{nm}SDCenterZero", (z["lb"] + z["ub"]) / 2, "coef")
    sdb = float(pd.read_csv(d / "honestdid_sd_breakdown.csv")["breakdown"].iloc[0])
    if np.isfinite(sdb):
        R.add(f"{nm}SDBreakdown", sdb, "num5")


def funnel_identification(D: Data, R: Registry, out: dict) -> None:
    """Event study, date placebos and HonestDiD for the funnel-qualified count,
    with the specification used for question volume."""
    m = D.main
    es, V, fit, _, _ = event_study(m, y="funnel")
    out["es_qual"], out["es_qual_V"] = es, V
    comp = es[es["complete"]]
    post = comp[comp["bin"] >= 0].sort_values("bin")
    pre = comp[comp["bin"] < -1]
    pre_terms, post_terms = [bin_name(int(b)) for b in pre["bin"]], [bin_name(int(b)) for b in post["bin"]]
    R.add("QualESPreMean", pre["beta"].mean(), "coef")
    R.add("QualESPostMean", post["beta"].mean(), "coef")
    R.add("QualESPostMin", post["beta"].min(), "coef")
    R.add("QualESPostMax", post["beta"].max(), "coef")
    out["es_qual_monotone"] = bool(np.all(np.diff(post["beta"].to_numpy()) < 0))
    ps = post["p"].to_numpy()
    first = next((i for i in range(len(ps)) if (ps[i:] < 0.05).all()), None)
    if first is not None:
        R.add("QualESFirstSigBin", int(post["bin"].iloc[first]), "int")
    wt = fit.wald_test(R=np.eye(len(fit.coef()))[[list(fit.coef().index).index(t) for t in pre_terms]])
    R.add("QualESPreJointP", float(wt["pvalue"]), "p")
    # date placebos on the pre-ChatGPT sample
    pm = m[m["week_start"] < CHATGPT]
    rows = [{"cutoff": c, **ddd(pm, y="funnel", cut=pd.Timestamp(c)).row()} for c in PLACEBO_CUTS]
    rows.append({"cutoff": COPILOT.strftime("%Y-%m-%d") + " (Copilot GA)", **ddd(pm, y="funnel", cut=COPILOT).row()})
    pl = pd.DataFrame(rows)
    out["placebos_qual"] = pl
    R.add("QualPlacMin", pl["beta"].min(), "scoef")
    R.add("QualPlacMax", pl["beta"].max(), "scoef")
    R.add("QualPlacPMin", pl["p"].min(), "pv")
    R.add("QualPlacPMax", pl["p"].max(), "pv")
    # HonestDiD on the funnel-qualified event study
    honestdid_run(R, "QualHD", es, V, pre_terms, post_terms, "honestdid_funnel", m["tag"].nunique() - 1)
    out["qual_terms"] = (pre_terms, post_terms)


def ddd_event_weights(df: pd.DataFrame, y: str) -> pd.Series:
    """Weights c_k with beta_DDD = sum_k c_k beta_k over the event-study bins
    (reference: bin -1), from the projection of the treatment with the Poisson
    weights of the pooled fit. The identity is exact in a linear model and holds
    up to the nonlinearity of PPML here; the post-period weights sum to one."""
    d = design(df)
    d["bin"] = event_bins(d["week_start"])
    d = d[d.groupby("tag_qtype")[y].transform("sum") > 0].reset_index(drop=True)
    d["mu"] = np.asarray(pf.fepois(f"{y} ~ ai_post + post_sub + aps | tag_qtype + week_id", data=d)
                         .predict(type="response"), dtype=float)
    xt = np.asarray(pf.feols("aps ~ ai_post + post_sub | tag_qtype + week_id", data=d, weights="mu").resid(),
                    dtype=float)
    den = float((d["mu"] * xt * d["aps"]).sum())
    c = pd.Series({int(k): float((d["mu"] * xt * d["ai"] * d["s"] * (d["bin"] == k)).sum() / den)
                   for k in sorted(d["bin"].unique()) if k != -1})
    del d
    gc.collect()
    return c


def beta_weighted_sensitivity(D: Data, R: Registry, out: dict) -> None:
    """HonestDiD for the combination of complete post-period event-study
    coefficients that carries the weights of beta_DDD, volume and funnel-qualified."""
    rows = []
    for y, nm, es, V in [("questions", "HDB", out["es"], out["es_V"]),
                         ("funnel", "QualHDB", out["es_qual"], out["es_qual_V"])]:
        c = ddd_event_weights(D.main, y)
        idx = es.set_index("bin")
        complete = idx[idx["complete"]].index
        pre_bins = sorted(int(b) for b in complete if b < -1)
        post_bins = sorted(int(b) for b in complete if b >= 0)
        partial = [b for b in c.index if b >= 0 and b not in post_bins]
        R.add(f"{nm}Repro", float((c * idx.loc[c.index, "beta"]).sum()), "coef")
        R.add(f"{nm}PartialShare", float(c[partial].sum()), "pct1")
        l = c[post_bins] / c[post_bins].sum()
        R.add(f"{nm}WeightFirst", float(l.iloc[0]), "pct0")
        R.add(f"{nm}WeightLast", float(l.iloc[-1]), "pct0")
        honestdid_run(R, nm, es, V, [bin_name(b) for b in pre_bins], [bin_name(b) for b in post_bins],
                      f"honestdid_beta_{y}", D.main["tag"].nunique() - 1, l.to_numpy())
        rows += [{"outcome": y, "bin": b, "weight": float(c[b]), "beta": float(idx.loc[b, "beta"])} for b in c.index]
    out["beta_event_weights"] = pd.DataFrame(rows)


def selenium_gap(D: Data, R: Registry, out: dict) -> None:
    """The collection gap of the selenium tag and the backfill of the following
    month: the estimates without the cells of the weeks the gap touches, without
    the backfilled pairs and without the tag."""
    m = D.main
    qm = D.q[D.q["week_start"].isin(D.weeks_main)]
    s = qm[qm["tag"] == "selenium"]
    days = pd.DatetimeIndex(sorted(s["day"].unique()))
    steps = pd.Series(days[1:] - days[:-1], index=days[:-1])
    start, end = steps.idxmax() + pd.Timedelta(days=1), steps.idxmax() + steps.max() - pd.Timedelta(days=1)
    other = qm[qm["tag"] != "selenium"].groupby("tag")["day"].agg(lambda d: pd.Series(sorted(d.unique())).diff().max())
    weeks = pd.date_range(start - pd.Timedelta(days=start.weekday()), end, freq="7D")
    bf = D.backfill
    rows = s[s["question_id"].isin(bf["question_id"])]
    R.add("SelGapStart", start.strftime("%Y-%m-%d"), "text")
    R.add("SelGapEnd", end.strftime("%Y-%m-%d"), "text")
    R.add("SelGapWeeks", len(weeks), "int")
    R.add("SelOtherGapDays", (other.max() - pd.Timedelta(days=1)).days, "int")
    R.add("SelBackfillPairs", len(rows), "int")
    R.add("SelBackfillStart", bf["creation_date"].min()[:10], "text")
    R.add("SelBackfillEnd", bf["creation_date"].max()[:10], "text")
    for y, nm, jack in [("questions", "Q", "jack_q"), ("funnel", "Qual", "jack_qual")]:
        R.est(f"SelDrop{nm}", ddd(m[~((m["tag"] == "selenium") & m["week_start"].isin(weeks))], y=y))
        R.est(f"SelNoBack{nm}", ddd(moved_panel(m, rows), y=y))
        R.add(f"SelNoTag{nm}B", out[jack]["selenium"], "coef")


def index_without_maturity(D: Data, R: Registry, out: dict) -> None:
    """Tag maturity varies for one tag only: its values, and the tag index built
    from the other three components."""
    ix = D.index.reset_index(drop=True)
    mat = ix["tag_maturity_weeks_pre"]
    odd = ix.index[mat != mat.max()]
    if len(odd) != 1:
        raise RuntimeError("tag maturity varies for more than one tag")
    if float((_zavg(ix, STRUCTURAL_COMPONENTS) - ix[AI]).abs().max()) > 1e-12:
        raise RuntimeError("the components do not reproduce the tag index")
    R.add("MaturityTag", ix.loc[odd[0], "tag"], "text")
    R.add("MaturityWeeksTag", mat[odd[0]], "int")
    R.add("MaturityWeeksOther", mat.max(), "int")
    R.add("MaturityNOther", int((mat == mat.max()).sum()), "int")
    R.add("MaturityZ", (mat[odd[0]] - mat.mean()) / mat.std(ddof=0), "num2")
    alt = _zavg(ix, [c for c in STRUCTURAL_COMPONENTS if c != "tag_maturity_weeks_pre"])
    delta = float(np.percentile(alt, 90) - np.percentile(alt, 10))
    m = D.main.assign(ai_nomat=D.main["tag"].map(dict(zip(ix["tag"], alt))))
    for y, nm in [("questions", "Q"), ("funnel", "Qual")]:
        e = ddd(m, y=y, ai="ai_nomat")
        R.est(f"NoMat{nm}", e)
        out[f"nomat_{nm.lower()}"] = e
        R.add(f"PropNoMat{nm}", np.expm1(e.b * delta), "spct1")
        R.add(f"PropNoMat{nm}Lo", np.expm1(e.lo * delta), "spct1")
        R.add(f"PropNoMat{nm}Hi", np.expm1(e.hi * delta), "spct1")


def placebos(D: Data, R: Registry, out: dict) -> None:
    m = D.main
    pre = m[m["week_start"] < CHATGPT]
    rows = []
    for cut, nm in zip(PLACEBO_CUTS, PLACEBO_NAMES):
        e = ddd(pre, cut=pd.Timestamp(cut))
        R.est(f"Plac{nm}", e, signed=True)
        rows.append({"cutoff": cut, **e.row()})
    cop = ddd(pre, cut=COPILOT)
    R.est("CopSolo", cop, signed=True)
    rows.append({"cutoff": COPILOT.strftime("%Y-%m-%d") + " (Copilot GA)", **cop.row()})
    pl = pd.DataFrame(rows)
    out["placebos"] = pl
    R.add("PlacMin", pl["beta"].min(), "scoef")
    R.add("PlacMax", pl["beta"].max(), "scoef")
    R.add("PlacPMax", pl["p"].max(), "pv")
    R.add("PlacPMin", pl["p"].min(), "pv")
    R.add("PlacN", len(pl), "int")
    R.add("CopilotDate", COPILOT.strftime("%Y-%m-%d"), "text")
    # joint model: ChatGPT and Copilot cutoffs on the full panel
    d = design(m)
    d["pc"] = (d["week_start"] >= COPILOT).astype(int)
    d["ai_pc"], d["pc_sub"], d["aps_c"] = d["ai"] * d["pc"], d["pc"] * d["s"], d["ai"] * d["pc"] * d["s"]
    jf = pf.fepois("questions ~ ai_post + post_sub + aps + ai_pc + pc_sub + aps_c + ai_sub | tag_qtype + week_id",
                   data=d, vcov={"CRV1": "tag"})
    R.est("JointGPT", est_from(jf, "aps"))
    R.est("JointCop", est_from(jf, "aps_c"), signed=True)
    out["joint"] = (est_from(jf, "aps"), est_from(jf, "aps_c"))
    # rolling placebos every four weeks
    cuts = pd.date_range("2020-07-01", "2022-09-01", freq="28D")
    roll = pd.DataFrame([{"cutoff": c, **ddd(pre, cut=c).row()} for c in cuts])
    out["rolling"] = roll
    R.add("RollN", len(roll), "int")
    R.add("RollMean", roll["beta"].mean(), "scoef")
    R.add("RollMin", roll["beta"].min(), "scoef")
    R.add("RollMax", roll["beta"].max(), "scoef")
    R.add("RollFirst", cuts.min().strftime("%Y-%m-%d"), "text")
    R.add("RollLast", cuts.max().strftime("%Y-%m-%d"), "text")
    R.add("RollBelowReal", int((roll["beta"] <= out["base"].b).sum()), "int")


def matched(D: Data, R: Registry, out: dict) -> None:
    m = D.main
    pre = m[m["week_start"] < CHATGPT]
    w = pre.groupby(["tag", "week_start"])["questions"].sum().reset_index()
    w["t"] = (w["week_start"] - w["week_start"].min()).dt.days // 7
    slope = w.groupby("tag").apply(lambda g: linregress(g["t"], np.log1p(g["questions"])).slope,
                                   include_groups=False)
    q = pd.qcut(slope, 4, labels=[1, 2, 3, 4]).astype(int)
    rows = []
    for k, nm in zip([1, 2, 3, 4], ["One", "Two", "Three", "Four"]):
        e = ddd(m[m["tag"].isin(q[q == k].index)])
        R.est(f"SlopeQ{nm}", e, signed=True)
        rows.append({"spec": f"slope quartile {k}", **e.row(), "n_tags": int((q == k).sum())})
    band = slope[(slope - slope.median()).abs() <= slope.quantile(0.75) - slope.quantile(0.25)].index
    out["band_tags"] = band
    e = ddd(m[m["tag"].isin(band)])
    R.est("Match", e)
    R.add("MatchNTags", len(band), "int")
    rows.append({"spec": "interquartile pre-slope band", **e.row(), "n_tags": len(band)})
    out["matched"] = pd.DataFrame(rows)


def trend_controls(D: Data, R: Registry, out: dict) -> None:
    """DDD with AI x Sub x t (and t squared) trend controls, t in weeks."""
    m = D.main.copy()
    t = ((m["week_start"] - m["week_start"].min()).dt.days // 7) / 52.0
    m["ais_t"] = m[AI] * m["sub"] * t
    m["ais_t2"] = m[AI] * m["sub"] * t ** 2
    out["trend_linear"] = ddd(m, extra=" + ais_t")
    out["trend_quadratic"] = ddd(m, extra=" + ais_t + ais_t2")
    R.est("LinTrend", out["trend_linear"])
    R.est("QuadTrend", out["trend_quadratic"])


def trend_extrapolation(D: Data, R: Registry, out: dict) -> None:
    """Differential AI x Sub trends estimated on the pre-period only and
    extrapolated: every post-period week gets its own AI x Sub dummy, so the
    trend terms are identified from pre-period variation alone; the post-period
    effect net of the extrapolated trend is the equal-weight mean of the
    post-week coefficients. The lower-order terms of eq. (1) are kept."""
    from scipy.stats import t as tdist
    m = design(D.main)
    weeks = ((m["week_start"] - m["week_start"].min()).dt.days // 7).astype(float)
    t = (weeks - weeks[m["post"] == 0].max()) / 52.0          # years; 0 = last pre-period week
    ais = m["ai"] * m["s"]
    post_weeks = sorted(m.loc[m["post"] == 1, "week_start"].unique())
    names = [f"pw{i}" for i in range(len(post_weeks))]
    m = pd.concat([m, pd.DataFrame({"ais_t": ais * t, "ais_t2": ais * t ** 2,
                                    **{n: ais * (m["week_start"] == w) for n, w in zip(names, post_weeks)}},
                                   index=m.index)], axis=1)
    df = m["tag"].nunique() - 1
    L = np.ones(len(names)) / len(names)
    rows = []
    for key, trend in [("TrendRef", ""), ("TrendLin", " + ais_t"), ("TrendQuad", " + ais_t + ais_t2")]:
        fit = pf.fepois(f"questions ~ ai_post + post_sub{trend} + {' + '.join(names)} | tag_qtype + week_id",
                        data=m, vcov={"CRV1": "tag"})
        co = fit.coef()
        V = pd.DataFrame(np.asarray(fit._vcov), index=co.index, columns=co.index)
        b = float(L @ co[names])
        se = float(np.sqrt(L @ V.loc[names, names].to_numpy() @ L))
        p = float(2 * tdist.sf(abs(b / se), df))
        R.add(f"{key}B", b, "coef")
        R.add(f"{key}SE", se, "coef")
        R.add(f"{key}P", p, "p")
        R.add(f"{key}Pv", p, "pv")
        row = {"spec": key, "net_post_effect": b, "se": se, "p": p}
        for k in ("ais_t", "ais_t2"):
            if k in co.index:
                row[f"{k}_coef"], row[f"{k}_se"] = float(co[k]), float(V.loc[k, k] ** 0.5)
        rows.append(row)
        if key == "TrendLin":
            R.add("PreTrendB", float(co["ais_t"]), "scoef")
            R.add("PreTrendSE", float(V.loc["ais_t", "ais_t"] ** 0.5), "coef")
        if key == "TrendQuad":
            # second difference of the quadratic trend over one 13-week bin, the unit of M in Delta-SD
            h = 13 / 52.0
            c = 2 * float(co["ais_t2"]) * h ** 2
            c_se = 2 * float(V.loc["ais_t2", "ais_t2"] ** 0.5) * h ** 2
            q = tdist.ppf(0.975, df)
            R.add("PreCurv", c, "coef4")
            R.add("PreCurvAbs", abs(c), "coef4")
            R.add("PreCurvLo", c - q * c_se, "coef4")
            R.add("PreCurvHi", c + q * c_se, "coef4")
            R.add("PreCurvBound", max(abs(c - q * c_se), abs(c + q * c_se)), "coef4")
            row.update(curvature_per_bin=c, curvature_se=c_se)
    out["trend_extrapolation"] = pd.DataFrame(rows)


def spec_curve(D: Data, R: Registry, out: dict) -> None:
    """Specification curve over sample x outcome x control-group choices,
    all with the structural index."""
    m = D.main
    vol, band = out["vol_rank"], out["band_tags"]
    samples = {"full": m, "top50": m[m["tag"].isin(vol.index[:50])],
               "top75": m[m["tag"].isin(vol.index[:75])], "iqr_band": m[m["tag"].isin(band)]}
    rows = []
    for sn, s in samples.items():
        for ctrl, sc in [("baseline", s), ("excl_AA", s[s["question_type"] != "advanced_architecture"])]:
            for y in ["questions", "unique_users"]:
                rows.append({"sample": sn, "control": ctrl, "outcome": y, **ddd(sc, y=y).row()})
    sc = pd.DataFrame(rows)
    out["spec_curve"] = sc
    R.add("SpecN", len(sc), "int")
    R.add("SpecNeg", int((sc["beta"] < 0).sum()), "int")
    R.add("SpecNegSig", int(((sc["beta"] < 0) & (sc["p"] < 0.05)).sum()), "int")
    R.add("SpecMin", sc["beta"].min(), "coef")
    R.add("SpecMax", sc["beta"].max(), "coef")


def late_confounders(D: Data, R: Registry, out: dict) -> None:
    m = D.main
    donut = m[~((m["week_start"] >= "2023-06-01") & (m["week_start"] < "2023-09-01"))]
    R.est("Donut", ddd(donut))
    R.add("DonutStart", "2023-06-01", "text")
    R.add("DonutEnd", "2023-08-31", "text")
    rows = [{"spec": "Full main window", **out["base"].row()},
            {"spec": "Excluding June-August 2023 (moderation strike)", **ddd(donut).row()}]
    for end, lab, nm in [("2023-06-05", "Window ends before the moderation strike (2023-06-05)", "PreStrike"),
                         ("2024-05-06", "Window ends before the SO-OpenAI licence (2024-05-06)", "PreLicence")]:
        e = ddd(m[m["week_start"] < end])
        R.est(nm, e)
        rows.append({"spec": lab, **e.row()})
    out["late"] = pd.DataFrame(rows)


def heterogeneity(D: Data, R: Registry, out: dict) -> None:
    """AI x Post within each question type (PPML, tag and week fixed effects)."""
    rows = []
    for col, label, _ in FUNNEL:
        for qt, nm in QTYPES:
            sub = design(D.main[D.main["question_type"] == qt])
            e = est_from(pf.fepois(f"{col} ~ ai_post | tag + week_id", data=sub, vcov={"CRV1": "tag"}), "ai_post")
            rows.append({"question_type": qt, "outcome": col, "label": label, **e.row(),
                         "substitutable": int(qt not in ("version_environment_specific", "advanced_architecture"))})
            if col == "questions":
                R.est(f"Het{nm}", e)
                if qt == "version_environment_specific":
                    R.add("HetVESMDE", mde80(e.se), "num2")
            gc.collect()
    het = pd.DataFrame(rows)
    out["het"] = het
    hq = het[(het["outcome"] == "questions") & (het["substitutable"] == 1)]
    R.add("HetSubMin", hq["beta"].min(), "coef")
    R.add("HetSubMax", hq["beta"].max(), "coef")


def magnitude(D: Data, R: Registry, out: dict) -> None:
    """Descriptive linear extrapolation of the pre-period log weekly volume."""
    wk = D.main.groupby("week_start")["questions"].sum().reset_index()
    wk["t"] = (wk["week_start"] - wk["week_start"].min()).dt.days // 7
    pre, post = wk[wk["week_start"] < CHATGPT], wk[wk["week_start"] >= CHATGPT].copy()
    fit = linregress(pre["t"], np.log(pre["questions"]))
    post["cf"] = np.exp(fit.intercept + fit.slope * post["t"])
    trend_short = float((post["cf"] - post["questions"]).sum())
    R.add("TrendShort", trend_short, "int")
    R.add("TrendShortM", trend_short / 1e6, "num2")
    R.add("TrendSlope", fit.slope, "num5")
    R.add("TrendRsq", fit.rvalue ** 2, "num2")
    R.add("TrendCFM", post["cf"].sum() / 1e6, "num2")
    R.add("ObsPostM", post["questions"].sum() / 1e6, "num2")
    R.add("TrendBreakPct", -trend_short / post["cf"].sum(), "spct0")
    out["counterfactual"] = (wk, post, fit)
