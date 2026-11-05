"""Data construction: one question-tag frame (2020-2025) with Monday-Sunday
weeks recomputed from the UTC creation timestamp, and every analysis panel
aggregated from it on the full grid of tags x complete weeks x question
types, with zero counts in cells that have no question. Only complete
Monday-Sunday weeks enter any panel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.prepare_stackoverflow_question_type_raw import (
    TAG_ALIASES, classify_questions, control_word_mask, source_files)
from src.data.prepare_answerability_index import (
    build_ai_answerability_from_aggregates, load_complexity)
from src.utils.io import read_csv

from .common import CHATGPT, PROCESSED, RAW_SO

COLS = ["tag", "question_id", "owner_user_id", "creation_date", "title",
        "body_length", "has_code", "score", "answer_count",
        "has_accepted_answer", "is_closed"]
NUM = ["question_id", "owner_user_id", "body_length", "has_code", "score",
       "answer_count", "has_accepted_answer", "is_closed"]
MAIN_LAST_DAY = pd.Timestamp("2024-12-31")
SCORE_BINS = ["s0", "s1", "s2", "s3", "s4", "s5p"]
INDEX_COLS = ["ai_answerability_structural", "ai_answerability_zscore",
              "ai_answerability_pca", "ai_answerability_quantile"]


@dataclass
class Data:
    q: pd.DataFrame          # question-tag rows, all complete weeks
    main: pd.DataFrame       # cell panel, main window
    ext: pd.DataFrame        # cell panel, main window + 2025
    index: pd.DataFrame      # tag-level treatment
    weeks_main: pd.DatetimeIndex
    weeks_ext: pd.DatetimeIndex
    dropped_weeks: pd.DataFrame
    index_window: tuple[pd.Timestamp, pd.Timestamp]
    # inputs reduced from the raw files; set by analysis_data.load
    counts: dict | None = None       # counts from the raw files, by name
    sample: pd.DataFrame | None = None        # classifier labels of the validation sample
    embed_tag: pd.DataFrame | None = None     # embedding answerability by tag
    type_scores: dict | None = None           # continuous substitutability by question type
    pool: pd.DataFrame | None = None          # revealed-answerability pool
    labels: pd.DataFrame | None = None        # MiniLM label of each panel question
    backfill: pd.DataFrame | None = None      # questions of the selenium backfill


def build_index() -> tuple[pd.DataFrame, tuple[pd.Timestamp, pd.Timestamp]]:
    """Tag-level AI-answerability indices from the SEDE tag-week extract,
    pre-ChatGPT weeks only (prepare_answerability_index)."""
    tw = read_csv(RAW_SO / "stackoverflow_tag_week.csv")
    tw["week_start"] = pd.to_datetime(tw["week_start"])
    ai = build_ai_answerability_from_aggregates(tw, load_complexity(RAW_SO))
    pre = tw[tw["week_start"] < CHATGPT]
    stored = pd.read_csv(PROCESSED / "ai_answerability_index.csv")
    chk = ai.merge(stored, on="tag", suffixes=("", "_stored"))
    for c in INDEX_COLS:
        if len(chk) != 100 or (chk[c] - chk[f"{c}_stored"]).abs().max() > 1e-9:
            raise RuntimeError(f"rebuilt {c} does not match ai_answerability_index.csv")
    return ai.reset_index(drop=True), (pre["week_start"].min(), pre["week_start"].max())


def load_questions(keep_source_week: bool = False, keep_title: bool = False) -> pd.DataFrame:
    extra = ["week_start"] if keep_source_week else []
    raw = pd.concat([pd.read_csv(f, usecols=COLS + extra)
                     for f in source_files(RAW_SO, year=None)], ignore_index=True)
    c25 = pd.read_csv(PROCESSED / "stackoverflow_2025_clean_question_tag.csv",
                      usecols=[c for c in COLS if c != "tag"] + ["tag_consulted"] + extra)
    c25 = c25.rename(columns={"tag_consulted": "tag"})
    q = pd.concat([raw, c25[COLS + extra]], ignore_index=True)
    if keep_source_week:
        q = q.rename(columns={"week_start": "week_source"})
        q["week_source"] = pd.to_datetime(q["week_source"], format="mixed")
    q["tag"] = q["tag"].replace(TAG_ALIASES)
    q = q.drop_duplicates(["question_id", "tag"]).reset_index(drop=True)
    for c in NUM:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["creation_date"] = pd.to_datetime(q["creation_date"], format="mixed")
    day = q["creation_date"].dt.normalize()
    q["week_start"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
    q = classify_questions(q)
    q["control_mask"] = control_word_mask(q["title"])
    if not keep_title:
        q = q.drop(columns=["title"])
    q["day"] = day
    acc = q["has_accepted_answer"].fillna(0) == 1
    closed = q["is_closed"].fillna(0) == 1
    score = q["score"].fillna(0)
    curated = acc & ~closed
    q["answered"] = (q["answer_count"].fillna(0) > 0).astype("int8")
    q["accepted"] = acc.astype("int8")
    q["acc_nonclosed"] = curated.astype("int8")
    q["acc_nonneg"] = (acc & (score >= 0)).astype("int8")
    q["funnel"] = (curated & (score >= 0)).astype("int8")
    for k in range(5):
        q[f"s{k}"] = (curated & (score == k)).astype("int8")
    q["s5p"] = (curated & (score >= 5)).astype("int8")
    q["value_w"] = np.where(curated, np.log1p(score.clip(lower=0)), 0.0)
    q["k_tags"] = q.groupby("question_id")["tag"].transform("size")
    return q


def complete_weeks(q: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DataFrame]:
    days = q.groupby("week_start")["day"].nunique()
    partial = days[days < 7]
    full = days[days == 7].index
    steps = pd.Series(full[1:] - full[:-1])
    if not (steps == pd.Timedelta(days=7)).all():
        raise RuntimeError("complete weeks are not contiguous")
    main = full[full <= MAIN_LAST_DAY]
    dropped = pd.DataFrame({"week_start": partial.index, "days": partial.values})
    return main, full, dropped


def score_ratio(q: pd.DataFrame) -> pd.Series:
    """Per-tag pre/post ratio of mean absolute score (1 where undefined)."""
    post = q["week_start"] >= CHATGPT
    a = q["score"].fillna(0).abs()
    m = a.groupby([q["tag"], post]).mean().unstack()
    return (m[False] / m[True].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def voting_normalised(q: pd.DataFrame, clip: tuple[float, float] | None,
                      ks=(1, 2, 5)) -> pd.DataFrame:
    """High-value flags after rescaling post-period scores by the tag's
    pre/post ratio of mean absolute score (clipped)."""
    post = q["week_start"] >= CHATGPT
    ratio = score_ratio(q)
    if clip is not None:
        ratio = ratio.clip(*clip)
    scale = np.where(post, q["tag"].map(ratio).to_numpy(), 1.0)
    s = q["score"].fillna(0).to_numpy() * scale
    cur = (q["acc_nonclosed"] == 1).to_numpy()
    return pd.DataFrame({f"vn{k}": (cur & (s >= k)).astype("int8") for k in ks},
                        index=q.index)


def cells(q: pd.DataFrame, index: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Cell panel on the full grid of index tags x weeks x question types.
    Every count, funnel stage and score bin is zero in a cell with no
    question; the cell means (body length, score) are missing there."""
    keys = ["tag", "week_start", "question_type"]
    sub = q.groupby("question_type")["substitutable_type"].agg(["min", "max"])
    if (sub["min"] != sub["max"]).any():
        raise RuntimeError("substitutability is not a function of the question type")
    if not set(q["tag"]).issubset(set(index["tag"])):
        raise RuntimeError("question tags outside the answerability index")
    w = q.assign(q_frac=1.0 / q["k_tags"])
    sums = ["answered", "accepted", "acc_nonclosed", "acc_nonneg", "funnel",
            *SCORE_BINS, "value_w", "q_frac", "answer_count", "is_closed",
            "has_code"] + [c for c in w.columns if c.startswith("vn")]
    g = w.groupby(keys)
    out = g[sums].sum()
    out["questions"] = g["question_id"].size()
    out["unique_users"] = g["owner_user_id"].nunique()
    out["body_length_mean"] = g["body_length"].mean()
    out["avg_score"] = g["score"].mean()
    grid = pd.MultiIndex.from_product([sorted(index["tag"]), weeks, sorted(sub.index)], names=keys)
    counts = [c for c in out.columns if c not in ("body_length_mean", "avg_score")]
    out = out.reindex(grid)
    out[counts] = out[counts].fillna(0)
    out = out.reset_index().rename(columns={"answer_count": "answers",
                                            "is_closed": "closed",
                                            "has_code": "code_q"})
    out["sub"] = out["question_type"].map(sub["min"]).astype(int)
    for k in range(6):
        out[f"ge{k}"] = out[SCORE_BINS[k:]].sum(axis=1)
    return finish_cells(out, index)


def finish_cells(out: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """The treatment measures and fixed-effect keys of a cell panel."""
    out = out.merge(index[["tag", *INDEX_COLS]], on="tag", how="left", validate="m:1")
    out["tag_qtype"] = out["tag"] + "::" + out["question_type"]
    out["week_id"] = out["week_start"].dt.strftime("%Y-%m-%d")
    return out.sort_values(["tag", "week_start", "question_type"]).reset_index(drop=True)


def build() -> Data:
    index, window = build_index()
    q = load_questions()
    weeks_main, weeks_ext, dropped = complete_weeks(q)
    q = q[q["week_start"].isin(weeks_ext)].reset_index(drop=True)
    qm = q[q["week_start"].isin(weeks_main)]
    vn = pd.concat([voting_normalised(qm, None).add_suffix("_noclip"),
                    voting_normalised(qm, (0.3, 5.0)).add_suffix("_c35"),
                    voting_normalised(qm, (0.5, 3.0)),
                    voting_normalised(qm, (0.7, 2.0)).add_suffix("_c72")], axis=1)
    main = cells(pd.concat([qm, vn], axis=1), index, weeks_main)
    ext = cells(q, index, weeks_ext)
    return Data(q, main, ext, index, weeks_main, weeks_ext, dropped, window)
