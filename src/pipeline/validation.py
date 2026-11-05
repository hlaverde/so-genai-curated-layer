"""Measurement validation: question-type classifier agreement, the
embedding-based answerability score, revealed answerability, the continuous
substitutability score, and the composition/resolvability mechanisms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from src.analysis.question_type_classifiers import SEEDS, SUBSTITUTABLE, fleiss_kappa
from src.data.prepare_stackoverflow_question_type_raw import CONTROL_WORDS, VERSION_ENV_WORDS

from .common import AI, CHATGPT, Registry, ddd, design, est_from
from .data import Data

EMB_PER_TAG = 1000


def classifier_agreement(D: Data, R: Registry, out: dict) -> None:
    s = D.sample.copy()
    for c in ["heuristic", "minilm", "mpnet"]:
        s[f"{c}_sub"] = s[f"{c}_type"].isin(SUBSTITUTABLE).astype(int)
    pairs = [("heuristic", "minilm"), ("heuristic", "mpnet"), ("minilm", "mpnet")]
    k7 = {p: cohen_kappa_score(s[f"{p[0]}_type"], s[f"{p[1]}_type"]) for p in pairs}
    kb = {p: cohen_kappa_score(s[f"{p[0]}_sub"], s[f"{p[1]}_sub"]) for p in pairs}
    rb = sum(np.column_stack([1 - s[f"{c}_sub"], s[f"{c}_sub"]]) for c in ["heuristic", "minilm", "mpnet"])
    cats = sorted(SEEDS)
    r7 = np.zeros((len(s), len(cats)), dtype=int)
    for c in ["heuristic_type", "minilm_type", "mpnet_type"]:
        for i, v in enumerate(s[c]):
            r7[i, cats.index(v)] += 1
    agree = ((s["heuristic_sub"] == s["minilm_sub"]) & (s["heuristic_sub"] == s["mpnet_sub"])).mean()
    R.add("ValN", len(s), "int")
    R.add("KappaEmbBin", kb[("minilm", "mpnet")], "num2")
    R.add("KappaEmbSeven", k7[("minilm", "mpnet")], "num2")
    R.add("KappaHeurMin", min(kb[pairs[0]], kb[pairs[1]]), "num2")
    R.add("KappaHeurMax", max(kb[pairs[0]], kb[pairs[1]]), "num2")
    R.add("FleissBin", fleiss_kappa(rb), "num2")
    R.add("FleissSeven", fleiss_kappa(r7), "num2")
    R.add("AgreeAll", agree, "pct1")
    R.add("NonSubHeur", 1 - s["heuristic_sub"].mean(), "pct1")
    emb_major = ((s["minilm_sub"] + s["mpnet_sub"]) >= 1)  # sub if any embedding says sub
    emb_non = ((1 - s["minilm_sub"]) + (1 - s["mpnet_sub"]) == 2)
    R.add("NonSubEmb", emb_non.mean(), "pct1")
    out["kappa"] = {"cohen7": k7, "cohen_bin": kb, "fleiss_bin": fleiss_kappa(rb),
                    "fleiss7": fleiss_kappa(r7), "agree": agree, "emb_sub_any": emb_major.mean()}
    out["validation_sample"] = s


def embedding_answerability(D: Data, R: Registry, out: dict) -> None:
    """Tag-level answerability from title similarity to fixed exemplars."""
    per_tag = D.embed_tag["n_titles"]
    tag = D.embed_tag[["tag", "embed"]].merge(D.index, on="tag", how="inner")
    tag["embed_z"] = (tag["embed"] - tag["embed"].mean()) / tag["embed"].std(ddof=0)
    r, p = pearsonr(tag["embed"], tag[AI])
    rs, ps = spearmanr(tag["embed"], tag[AI])
    R.add("EmbN", per_tag.sum(), "int")
    R.add("EmbPerTagMax", EMB_PER_TAG, "int")
    R.add("EmbPerTagMin", per_tag.min(), "int")
    R.add("EmbPerTagMedian", per_tag.median(), "int")
    R.add("EmbNTags", len(tag), "int")
    R.add("EmbCorrR", r, "num2")
    R.add("EmbCorrP", p, "p")
    R.add("EmbCorrRho", rs, "num2")
    out["embed_corr"] = pd.DataFrame([
        {"measure": k, "pearson": pearsonr(tag["embed"], tag[k])[0], "pearson_p": pearsonr(tag["embed"], tag[k])[1],
         "spearman": spearmanr(tag["embed"], tag[k])[0]}
        for k in ["ai_answerability_structural", "ai_answerability_zscore",
                  "ai_answerability_pca", "ai_answerability_quantile"]])
    # embedding score as the treatment intensity in eq. (1)
    m = D.main.merge(tag[["tag", "embed_z"]], on="tag", how="inner")
    e = ddd(m, ai="embed_z")
    # the inference of Table 4 with the embedding score as the treatment intensity
    from .estimates import jackknife, leave_out_macros, score_bootstrap
    eq = ddd(m, y="funnel", ai="embed_z")
    R.est("EmbQual", eq)
    for nm, y, est in [("EmbQ", "questions", e), ("EmbQual", "funnel", eq)]:
        p = score_bootstrap(m, y, ai="embed_z")
        R.add(f"KS{nm}P", p, "p")
        R.add(f"KS{nm}Pv", p, "pv")
        cr3, bj = jackknife(m, y, est, ai="embed_z")
        R.est(f"{nm}CRThree", cr3)
        leave_out_macros(R, nm, bj, est.b)
    R.est("EmbDDD", e)
    out["embed_ddd"] = e
    # partial regressions of the embedding score on the structural index
    tag["log_pre_volume"] = np.log1p(tag["questions_pre"])
    rows = []
    for label, x in [("(1) bivariate", []), ("(2) + log pre-volume", ["log_pre_volume"]),
                     ("(3) + tag maturity", ["log_pre_volume", "tag_maturity_weeks_pre"]),
                     ("(4) + accepted-answer rate", ["log_pre_volume", "tag_maturity_weeks_pre",
                                                     "accepted_answer_rate_pre"])]:
        f = sm.OLS(tag["embed"], sm.add_constant(tag[[AI] + x])).fit(cov_type="HC1")
        rows.append({"spec": label, "beta": f.params[AI], "se": f.bse[AI], "p": f.pvalues[AI],
                     "r2": f.rsquared, "n": int(f.nobs)})
    out["partial"] = pd.DataFrame(rows)
    R.add("PartialFullP", rows[-1]["p"], "p")


def revealed(D: Data, R: Registry, out: dict) -> None:
    df = D.pool.merge(D.index[["tag", AI]], on="tag", how="inner")
    df = df.sort_values(["question_id", AI], ascending=[True, False]).drop_duplicates("question_id")
    minutes = pd.to_numeric(df["minutes_to_first_answer"], errors="coerce")
    acc = pd.to_numeric(df["has_accepted_answer"], errors="coerce").fillna(0).astype(int)
    answered = (pd.to_numeric(df["answer_count"], errors="coerce").fillna(0) > 0).astype(int)
    not_closed = 1 - pd.to_numeric(df["is_closed"], errors="coerce").fillna(0).astype(int)
    fast = ((minutes <= 24 * 60) & minutes.notna()).astype(int)
    inv = -np.log1p(minutes.clip(lower=0))
    comp = pd.DataFrame({"accepted": acc, "answered_not_closed": answered * not_closed,
                         "accepted_fast_24h": acc * fast, "fast_answer_24h": fast,
                         "inverse_log_minutes": inv.fillna(inv.min())})
    z = comp.apply(lambda c: (c - c.mean()) / c.std(ddof=0) if c.std(ddof=0) > 0 else c * 0)
    df["revealed"] = z.mean(axis=1)
    df["high"] = (df["revealed"] >= df["revealed"].quantile(0.67)).astype(int)
    tag = df.groupby("tag").agg(revealed=("revealed", "mean"), ai=(AI, "first"))
    r, p = pearsonr(tag["ai"], tag["revealed"])
    R.add("RevealN", len(df), "int")
    R.add("RevealR", r, "num3")
    R.add("RevealP", p, "p")
    R.add("RevealRho", spearmanr(tag["ai"], tag["revealed"])[0], "num3")
    auc = roc_auc_score(df["high"], df[AI])
    R.add("RevealAUC", auc, "num2")
    df["qrt"] = pd.qcut(df[AI].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rows = [{"stratum": "Overall", "auc": auc, "n": len(df)}]
    for k in range(1, 5):
        s = df[df["qrt"] == k]
        rows.append({"stratum": f"Tag-index quartile {k}",
                     "auc": roc_auc_score(s["high"], s[AI]) if s["high"].nunique() > 1 else np.nan,
                     "n": len(s)})
    out["auc_strata"] = pd.DataFrame(rows)
    out["revealed_tag"] = (r, p, spearmanr(tag["ai"], tag["revealed"])[0], auc, len(df))


def continuous_substitutability(D: Data, R: Registry, out: dict) -> None:
    scores = D.type_scores
    m = D.main.assign(sc=D.main["question_type"].map(scores))
    m["sc"] = (m["sc"] - m["sc"].mean()) / m["sc"].std(ddof=0)
    R.est("ContSub", ddd(m, sub="sc"))
    out["cont_scores"] = scores


def keyword_sensitivity(D: Data, R: Registry, out: dict) -> None:
    """Leave-one-keyword-out on the title keywords of the two control types. A title
    whose only control keyword is the one left out joins the treated types; which
    of them it joins does not change the estimate of eq. (1), which depends on the
    question types only through Sub (Poisson is collapsible over them)."""
    from .estimates import moved_panel
    qm = D.q.loc[D.q["week_start"].isin(D.weeks_main),
                 ["tag", "week_start", "question_type", "question_id", "funnel", "control_mask"]]
    m = D.main[["tag", "week_start", "question_type", "questions", "funnel", "sub", AI, "tag_qtype", "week_id"]]
    rows = []
    for j, word in enumerate(CONTROL_WORDS):
        moved = qm[qm["control_mask"] == (1 << j)]
        p = moved_panel(m, moved, "other_conceptual")
        eq, ef = ddd(p, y="questions"), ddd(p, y="funnel")
        rows.append({"keyword": word, "list": "VES" if word in VERSION_ENV_WORDS else "AA", "pairs": len(moved),
                     "q_beta": eq.b, "q_se": eq.se, "q_p": eq.p, "f_beta": ef.b, "f_se": ef.se, "f_p": ef.p})
    d = pd.DataFrame(rows)
    out["keyword_sensitivity"] = d
    R.add("LokoN", len(d), "int")
    for c, nm in [("q", "Q"), ("f", "Qual")]:
        R.add(f"Loko{nm}Min", d[f"{c}_beta"].min(), "coef")
        R.add(f"Loko{nm}Max", d[f"{c}_beta"].max(), "coef")
        R.add(f"Loko{nm}MinWord", d.loc[d[f"{c}_beta"].idxmin(), "keyword"], "text")
        R.add(f"Loko{nm}MaxWord", d.loc[d[f"{c}_beta"].idxmax(), "keyword"], "text")
        R.add(f"Loko{nm}PMax", d[f"{c}_p"].max(), "pv")


def minilm_panel(D: Data, R: Registry, out: dict) -> None:
    """Eq. (1) with Sub taken from the MiniLM nearest-centroid label of every title
    of the main panel (src/analysis/classify_panel_titles.py): CR1, CR3 and the
    score bootstrap, for question volume and funnel-qualified posts."""
    from .estimates import jackknife, score_bootstrap
    labels = D.labels
    q = D.q.loc[D.q["week_start"].isin(D.weeks_main), ["tag", "week_start", "question_id", "funnel", "substitutable_type"]]
    q = q.merge(labels, on="question_id", how="left", validate="m:1")
    if q["minilm_type"].isna().any():
        raise RuntimeError("panel questions without a MiniLM label")
    q["minilm_sub"] = q["minilm_type"].isin(SUBSTITUTABLE).astype(int)
    keys = ["tag", "week_start", "minilm_type"]
    c = q.groupby(keys).agg(questions=("question_id", "size"), funnel=("funnel", "sum"))
    grid = pd.MultiIndex.from_product([sorted(D.index["tag"]), D.weeks_main, sorted(SEEDS)], names=keys)
    c = c.reindex(grid, fill_value=0).reset_index()
    c["sub"] = c["minilm_type"].isin(SUBSTITUTABLE).astype(int)
    c = c.merge(D.index[["tag", AI]], on="tag", how="left", validate="m:1")
    c["tag_qtype"] = c["tag"] + "::" + c["minilm_type"]
    c["week_id"] = c["week_start"].dt.strftime("%Y-%m-%d")
    R.add("MiniLMNTitles", len(labels), "int")
    R.add("MiniLMNonSub", 1 - q["minilm_sub"].mean(), "pct1")
    R.add("HeurNonSub", 1 - q["substitutable_type"].mean(), "pct1")
    R.add("MiniLMAgree", (q["minilm_sub"] == q["substitutable_type"]).mean(), "pct1")
    R.add("MiniLMKappa", cohen_kappa_score(q["minilm_sub"], q["substitutable_type"]), "num2")
    for y, nm in [("questions", "Q"), ("funnel", "Qual")]:
        e = ddd(c, y=y)
        R.est(f"MiniLM{nm}", e)
        cr3, _ = jackknife(c, y, e)
        R.est(f"MiniLM{nm}CRThree", cr3)
        p = score_bootstrap(c, y)
        R.add(f"KSMiniLM{nm}P", p, "p")
        R.add(f"KSMiniLM{nm}Pv", p, "pv")
    out["minilm_cells"] = c.groupby(["tag", "week_start", "sub"], as_index=False)[["questions", "funnel"]].sum()


def mechanisms(D: Data, R: Registry, out: dict) -> None:
    m = D.main[D.main["questions"] > 0]          # shares and cell means exist only where there are questions
    # composition: tag-week means, Post and Post x AI
    w = (m.assign(acc_share=m["accepted"] / m["questions"], code_share=m["code_q"] / m["questions"])
         .groupby(["tag", "week_start"])
         .agg(body_length=("body_length_mean", "mean"), accepted_share=("acc_share", "mean"),
              code_share=("code_share", "mean")).reset_index()
         .merge(D.index[["tag", AI]], on="tag"))
    w["post"] = (w["week_start"] >= CHATGPT).astype(int)
    w["post_ai"] = w["post"] * w[AI]
    rows = []
    for y, nm in [("body_length", "Body"), ("accepted_share", "AccShare"), ("code_share", "Code")]:
        a = est_from(pf.feols(f"{y} ~ post | tag", data=w, vcov={"CRV1": "tag"}), "post")
        b = est_from(pf.feols(f"{y} ~ post + post_ai | tag", data=w, vcov={"CRV1": "tag"}), "post_ai")
        R.est(f"Comp{nm}", a, signed=True)
        if y == "body_length":
            R.add("CompBodyChars", a.b, "sint")
        R.est(f"Comp{nm}AI", b, signed=True)
        rows.append({"outcome": y, "post": a.b, "post_p": a.p, "post_ai": b.b, "post_ai_p": b.p})
    comp = pd.DataFrame(rows)
    out["complementarity"] = comp
    R.add("CompAIPMin", comp["post_ai_p"].min(), "pv")
    R.add("CompAIPMax", comp["post_ai_p"].max(), "pv")
    # resolvability and quality shares: weighted DDD at the cell level
    r = m.assign(accepted_share=m["accepted"] / m["questions"], answered_share=m["answered"] / m["questions"],
                 closed_share=m["closed"] / m["questions"], mean_answer_count=m["answers"] / m["questions"])
    r["unanswered_share"], r["no_accepted_share"] = 1 - r["answered_share"], 1 - r["accepted_share"]
    res = []
    for y, label, nm in [("accepted_share", "Accepted share", "ResAcc"), ("answered_share", "Answered share", "ResAns"),
                         ("unanswered_share", "Unanswered share", "ResUnans"),
                         ("no_accepted_share", "No-accepted share", "ResNoAcc"),
                         ("closed_share", "Closed share", "ResClosed"),
                         ("mean_answer_count", "Mean answer count", "ResAnswers")]:
        e = ddd(r, y=y, model="ols", weights="questions")       # shares, not counts: weighted OLS
        R.est(nm, e, signed=True)
        res.append({"outcome": y, "label": label, **e.row()})
    out["resolvability"] = pd.DataFrame(res)
    # pre/post means by AI group (median split of the 100 tags), question-weighted
    med = D.index[AI].median()
    r["group"] = np.where(r[AI] > med, "HighAI", "LowAI")
    r["period"] = np.where(r["week_start"] < CHATGPT, "Pre", "Post")
    r["score_sum"] = r["avg_score"] * r["questions"]
    r["body_sum"] = r["body_length_mean"] * r["questions"]
    g = r.groupby(["group", "period"])[["questions", "body_sum", "accepted", "answered", "answers",
                                         "score_sum", "closed", "code_q"]].sum()
    tab = pd.DataFrame({
        "Body length (chars)": g["body_sum"] / g["questions"],
        "Accepted share": g["accepted"] / g["questions"],
        "Answered share": g["answered"] / g["questions"],
        "Mean answer count": g["answers"] / g["questions"],
        "Mean score": g["score_sum"] / g["questions"],
        "Closed share": g["closed"] / g["questions"],
        "Code share": g["code_q"] / g["questions"]})
    out["prepost_groups"] = tab
