"""2025 extension on the full 100-tag panel and the monitoring procedure."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .common import AI, CHATGPT, SEED, Registry, ddd, mde80  # noqa: F401
from .data import Data
from .estimates import event_study

YEAR24 = (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"))


def extension(D: Data, R: Registry, out: dict) -> None:
    ext = D.ext
    R.add("NWeeksExt", len(D.weeks_ext), "int")
    R.add("ExtLastDay", (D.weeks_ext.max() + pd.Timedelta(days=6)).strftime("%Y-%m-%d"), "text")
    R.add("ExtFirstWeek", D.weeks_ext[D.weeks_ext > D.weeks_main.max()].min().strftime("%Y-%m-%d"), "text")
    R.add("NCellsExt", len(ext), "int")
    R.add("CellsEmptyExt", (ext["questions"] == 0).mean(), "pct1")
    y25 = ext[~ext["week_start"].isin(D.weeks_main)]
    R.add("NCellsTwentyFive", len(y25), "int")
    R.add("CellsEmptyTwentyFive", (y25["questions"] == 0).mean(), "pct1")
    q25 = D.q[D.q["creation_date"] >= "2025-01-01"]
    R.add("NUniqueTwentyFive", q25["question_id"].nunique(), "int")
    R.add("NCollectedTwentyFive", int(D.counts["NCollectedTwentyFive"]), "int")
    # the main-window portion of the extended panel is the main panel itself
    gate = ext[ext["week_start"].isin(D.weeks_main)]
    k = ["tag", "week_start", "question_type"]
    same = gate[k + ["questions"]].merge(D.main[k + ["questions"]], on=k, how="outer", indicator=True)
    out["gate_identical"] = bool((same["_merge"] == "both").all()
                                 and (same["questions_x"] == same["questions_y"]).all())
    g = ddd(gate)
    R.est("Gate", g)
    out["gate"] = g
    full = ddd(ext)
    R.est("ExtFull", full)
    R.est("ExtNoAA", ddd(ext[ext["question_type"] != "advanced_architecture"]))
    R.est("ExtFunQual", ddd(ext, y="funnel"))
    es, _, _, _, _ = event_study(ext)
    out["es_ext"] = es
    post = es[(es["bin"] >= 0) & es["complete"]].sort_values("bin")
    out["es_ext_monotone"] = bool(np.all(np.diff(post["beta"].to_numpy()) < 0))
    for _, r in post.iterrows():
        nm = "ExtES" + {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
                        7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven"}[int(r["bin"])]
        R.add(nm, r["beta"], "coef")
    R.add("ExtESLastBin", int(post["bin"].max()), "int")
    R.add("ExtESNPost", len(post), "int")
    R.add("ExtESLastEnd", post["last_day"].max().strftime("%Y-%m-%d"), "text")
    R.add("ExtESMin", post["beta"].min(), "coef")


def tag_period(D: Data) -> pd.DataFrame:
    """Weekly means per tag: pre-ChatGPT, calendar 2024 and 2025 weeks."""
    e = D.ext
    per = np.where(e["week_start"] < CHATGPT, "base",
                   np.where(e["week_start"] < YEAR24[0], "y23",
                            np.where(e["week_start"].isin(D.weeks_main), "y24", "y25")))
    g = (e.assign(per=per).groupby(["tag", "per"])
         .agg(q=("questions", "sum"), hv=("ge1", "sum"), nwk=("week_start", "nunique")).reset_index())
    g["q_wk"], g["hv_wk"] = g["q"] / g["nwk"], g["hv"] / g["nwk"]
    w = g.pivot(index="tag", columns="per", values=["q_wk", "hv_wk"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w


def monitor(D: Data, R: Registry, out: dict) -> None:
    w = tag_period(D)
    w["drop24_act"] = 1 - w["q_wk_y24"] / w["q_wk_base"]
    w["drop24_hv"] = 1 - w["hv_wk_y24"] / w["hv_wk_base"]
    w["drop25_hv"] = 1 - w["hv_wk_y25"] / w["hv_wk_base"]
    v = w[w["drop24_act"] > 0.05].copy()
    v["vtd24"] = v["drop24_hv"] / v["drop24_act"]
    v["eroding"] = (v["drop25_hv"] >= v["drop25_hv"].quantile(2 / 3)).astype(int)
    R.add("AUCAct", roc_auc_score(v["eroding"], v["drop24_act"]), "num2")
    R.add("AUCVtD", roc_auc_score(v["eroding"], v["vtd24"]), "num2")
    R.add("MonitorNTags", len(v), "int")
    rho, p = spearmanr(v["drop24_hv"], v["drop25_hv"])
    R.add("PersistRho", rho, "num2")
    R.add("PersistP", p, "p")
    out["monitor_tags"] = v
    # 95% intervals from a bootstrap over tags (the target label of each tag is kept fixed)
    gen = np.random.default_rng(SEED + 1)
    boot = []
    while len(boot) < 999:
        s = v.iloc[gen.integers(0, len(v), len(v))]
        if s["eroding"].nunique() < 2:
            continue
        boot.append((roc_auc_score(s["eroding"], s["drop24_act"]), roc_auc_score(s["eroding"], s["vtd24"]),
                     spearmanr(s["drop24_hv"], s["drop25_hv"])[0]))
    b = np.array(boot)
    for j, nm in enumerate(["AUCAct", "AUCVtD", "PersistRho"]):
        R.add(f"{nm}Lo", np.percentile(b[:, j], 2.5), "num2")
        R.add(f"{nm}Hi", np.percentile(b[:, j], 97.5), "num2")
    R.add("MonitorBootReps", len(b), "int")
    out["monitor_boot"] = pd.DataFrame(b, columns=["auc_act", "auc_vtd", "persist_rho"])

    # intensive margin: budgeted allocation, out of sample on realised 2025 loss
    d = pd.DataFrame({"cost": w["q_wk_y24"],
                      "naive": (w["q_wk_base"] - w["q_wk_y24"]).clip(lower=0),
                      "calibrated": (w["hv_wk_base"] - w["hv_wk_y24"]).clip(lower=0),
                      "value": (w["hv_wk_base"] - w["hv_wk_y25"]).clip(lower=0)}).dropna()
    d = d[(w.loc[d.index, "hv_wk_base"] > 0) & (w.loc[d.index, "q_wk_base"] > 0)]

    def frac(key, b):
        x = d.assign(eff=d[key] / d["cost"]).sort_values("eff", ascending=False)
        cap, spent, val = b * d["cost"].sum(), 0.0, 0.0
        for _, r in x.iterrows():
            if spent + r["cost"] <= cap:
                spent += r["cost"]
                val += r["value"]
            else:
                val += max(cap - spent, 0) / r["cost"] * r["value"]
                break
        return val / d["value"].sum()

    rng = np.random.default_rng(SEED)
    rows = []
    for b in (0.10, 0.25, 0.40, 0.50):
        orc, cal, nai = frac("value", b), frac("calibrated", b), frac("naive", b)
        rs = []
        for _ in range(1500):
            x = d.sample(frac=1.0, random_state=int(rng.integers(1e9)))
            cc = x["cost"].cumsum() / d["cost"].sum()
            rs.append(x.loc[cc <= b, "value"].sum() / d["value"].sum())
        rows.append({"budget": b, "oracle": orc, "calibrated": cal, "naive": nai, "random": float(np.mean(rs)),
                     "regret_calibrated": (orc - cal) / orc, "regret_naive": (orc - nai) / orc})
    voi = pd.DataFrame(rows)
    out["voi"] = voi
    R.add("RegretNaiveMin", voi["regret_naive"].min(), "pct0")
    R.add("RegretNaiveMax", voi["regret_naive"].max(), "pct0")
    R.add("RegretCalibMax", voi["regret_calibrated"].max(), "pct1")
    R.add("VOINTags", len(d), "int")

    # value-disproportionality monitor on the main post period
    m = D.main[D.main["week_start"] >= CHATGPT]
    g = (m[AI] - out["reference"]["ai"]) * m["sub"]
    b_act, b_hv = out["base"].b, out["value_weighted"].loc[1, "beta"]
    by = (m.assign(da=m["questions"] * (np.exp(-b_act * g) - 1), dh=m["ge1"] * (np.exp(-b_hv * g) - 1))
          .groupby("tag")[["da", "dh"]].sum())
    by = by[by["da"] > 0].copy()
    by["vtd"] = by["dh"] / by["da"]
    # benchmark on the same scale: the ratio for all plotted tags pooled
    theta = by["dh"].sum() / by["da"].sum()
    by["flag"] = by["vtd"] > theta
    by["rank_act"] = by["da"].rank(ascending=False)
    rho2, p2 = spearmanr(by["rank_act"], by["vtd"].rank(ascending=False))
    R.add("VtDTheta", theta, "num3")
    R.add("VtDRho", rho2, "num2")
    R.add("VtDRhoP", p2, "p")
    R.add("VtDNTags", len(by), "int")
    R.add("VtDNAbove", int(by["flag"].sum()), "int")
    out["vtd"] = (by, theta)
