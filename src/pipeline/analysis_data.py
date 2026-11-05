"""The analysis data in data/analysis/: every input of the estimation, with no
question text. write_from_raw() builds them from the raw files, reducing what
depends on titles to labels and scores, and load() reads them. Every run of the
estimation starts from load()."""
from __future__ import annotations

import gc
import html

import numpy as np
import pandas as pd

from src.analysis.question_type_classifiers import SEED as CLASSIFIER_SEED
from src.analysis.question_type_classifiers import embed_and_assign, load_stratified_sample
from src.data.prepare_stackoverflow_question_type_raw import NON_SUBSTITUTABLE, TAG_ALIASES, classify_questions
from src.models.embedding_substitutability import build_substitutability_score_from_embeddings

from .common import ANALYSIS, CHATGPT, PROCESSED, RAW_SO
from .data import Data, INDEX_COLS, build, finish_cells

GZIP = {"method": "gzip", "compresslevel": 9, "mtime": 0}  # same bytes on every run
MAX_FILE_BYTES = 100_000_000
LABELS = "minilm_panel_title_labels"
QUESTION_COLS = ["question_id", "tag", "day", "question_type", "control_mask", "score", "curated"]
MINILM, MPNET = "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"

EMB_SEED, EMB_PER_TAG = 20260527, 1000
EXEMPLARS_EASY = [
    "How do I reverse a string in Python?",
    "What is the syntax for a for-loop in JavaScript?",
    "How do I parse JSON in Java?",
    "How can I sort a list of dictionaries by a key in Python?",
    "How do I write a basic React functional component?",
    "What is the difference between map and forEach in JavaScript?",
]
EXEMPLARS_HARD = [
    "My company-specific build pipeline crashes with code 137 only on "
    "our internal CI runner after Kubernetes 1.27 upgrade",
    "Why does our custom CUDA kernel deadlock on our A100 when launched "
    "from PyTorch 2.0.1 in our private cluster",
    "Our legacy authentication microservice returns 503 only when "
    "called from container deployed to our Tencent Cloud production "
    "environment with cilium CNI",
    "Specific version conflict between our internal protobuf fork and "
    "grpc 1.58 on RHEL 8 with FIPS mode enabled",
    "Architecture decision: should we use saga or two-phase commit for "
    "our specific event-sourced inventory system with current Kafka "
    "topology",
    "Compiler error on private toolchain when cross-compiling proprietary "
    "firmware for our internal RISC-V SoC under Yocto Honister",
]


# --------------------------------------------------------------------------
# reductions of the raw files
# --------------------------------------------------------------------------
def source_validation_counts() -> dict[str, int]:
    """API pull versus SEDE extract for the same day (2023-05-02)."""
    api = pd.read_csv(PROCESSED / "_validation" / "stackoverflow_question_type_raw_2023-05-02_2023-05-03.csv")
    sede = pd.read_csv(RAW_SO / "stackoverflow_question_type_raw_2023-05-01_2023-05-05.csv")
    for x in (api, sede):
        x["tag"] = x["tag"].replace(TAG_ALIASES)
    sede = sede[sede["creation_date"].str.startswith("2023-05-02") & sede["tag"].isin(api["tag"].unique())]
    key = ["question_id", "tag"]
    j = api.merge(sede, on=key, how="outer", suffixes=("_a", "_s"), indicator=True)
    b = j[j["_merge"] == "both"].copy()

    def differs(c):
        x, y = b[f"{c}_a"], b[f"{c}_s"]
        return ~((x == y) | (x.isna() & y.isna()))

    other = sum(int(differs(c).sum()) for c in ["owner_user_id", "creation_date", "has_code", "score",
                                                 "answer_count", "has_accepted_answer", "is_closed"])
    unesc = b["title_a"].fillna("").map(html.unescape) != b["title_s"].fillna("")
    types_a = classify_questions(b.rename(columns={"title_a": "title", "body_length_a": "body_length",
                                                   "has_code_a": "has_code"}))["question_type"]
    types_s = classify_questions(b.rename(columns={"title_s": "title", "body_length_s": "body_length",
                                                   "has_code_s": "has_code"}))["question_type"]
    return {"ValAPIRows": len(api), "ValSEDERows": len(sede), "ValUnmatched": int((j["_merge"] != "both").sum()),
            "ValTitleEscaped": int((differs("title") & ~unesc).sum()), "ValTitleOther": int(unesc.sum()),
            "ValBodyDiff": int(differs("body_length").sum()), "ValOtherDiff": other,
            "ValTypeDiff": int((types_a.to_numpy() != types_s.to_numpy()).sum())}


def classifier_sample() -> pd.DataFrame:
    """The stratified validation sample with the labels of the three classifiers."""
    s = load_stratified_sample(RAW_SO, 10000, CLASSIFIER_SEED)
    titles = s["title"].tolist()
    s["minilm_type"] = embed_and_assign(titles, MINILM)
    s["mpnet_type"] = embed_and_assign(titles, MPNET)
    return s


def embedding_tag_scores(index: pd.DataFrame) -> pd.DataFrame:
    """Tag-level answerability: the similarity of up to EMB_PER_TAG random
    pre-ChatGPT titles to fixed exemplars, averaged by tag."""
    from sentence_transformers import SentenceTransformer
    files = []
    for f in sorted(RAW_SO.glob("stackoverflow_question_type_raw_*.csv")):
        try:
            if pd.Timestamp(f.stem.split("_")[-1]) < CHATGPT:
                files.append(f)
        except ValueError:
            files.append(f)
    raw = pd.concat([pd.read_csv(f, usecols=["tag", "week_start", "question_id", "title"]) for f in files],
                    ignore_index=True)
    raw["tag"] = raw["tag"].replace(TAG_ALIASES)
    raw["week_start"] = pd.to_datetime(raw["week_start"], errors="coerce", format="mixed")
    raw = raw[(raw["week_start"] < CHATGPT) & raw["tag"].isin(index["tag"])].dropna(subset=["title"])
    raw = raw[raw["title"].str.len() > 5].drop_duplicates(["question_id", "tag"])
    raw = raw.sort_values(["tag", "question_id"])
    sample = (raw.groupby("tag", group_keys=False)[["tag", "question_id", "title"]]
              .apply(lambda g: g.sample(n=min(EMB_PER_TAG, len(g)), random_state=EMB_SEED))
              .reset_index(drop=True))
    del raw
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda t: model.encode(t, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    qe = enc(sample["title"].astype(str).tolist())
    sample["score"] = (qe @ enc(EXEMPLARS_EASY).T).mean(axis=1) - (qe @ enc(EXEMPLARS_HARD).T).mean(axis=1)
    return sample.groupby("tag").agg(embed=("score", "mean"), n_titles=("score", "size")).reset_index()


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def _write(df: pd.DataFrame, name: str) -> None:
    path = ANALYSIS / f"{name}.csv.gz"
    df.to_csv(path, index=False, compression=GZIP)
    if path.stat().st_size >= MAX_FILE_BYTES:
        raise RuntimeError(f"{path.name} reaches the 100 MB limit of a file")


def write_from_raw() -> None:
    labels = ANALYSIS / f"{LABELS}.csv.gz"
    if not labels.exists():
        raise FileNotFoundError(f"{labels} is missing: run python -m src.analysis.classify_panel_titles")
    D = build()
    print(f"[raw] {len(D.main):,} cells main, {len(D.ext):,} cells extended", flush=True)
    derived = [c for c in D.main.columns if c in INDEX_COLS or c in ("tag_qtype", "week_id")]
    _write(D.main.drop(columns=derived), "panel_main")
    _write(D.ext.drop(columns=derived), "panel_extended")
    _write(D.index, "tag_index")
    _write(pd.DataFrame({"week_start": D.weeks_ext, "main_window": D.weeks_ext.isin(D.weeks_main).astype(int)}), "weeks")
    _write(D.dropped_weeks, "dropped_weeks")
    q = D.q.assign(curated=lambda x: ((x["has_accepted_answer"].fillna(0) == 1)
                                      & ~(x["is_closed"].fillna(0) == 1)).astype("int8"))[QUESTION_COLS]
    _write(q, "question_tag")
    del q
    gc.collect()
    print("[raw] panels and question-tag rows written", flush=True)
    s = classifier_sample()
    _write(s.drop(columns=["title"]), "validation_sample")
    scores = build_substitutability_score_from_embeddings(D.main, s, MINILM)
    _write(pd.DataFrame({"question_type": list(scores), "score": list(scores.values())}), "question_type_substitutability")
    print("[raw] classifier sample and type scores written", flush=True)
    index, window = D.index, D.index_window
    del D, s
    gc.collect()
    _write(embedding_tag_scores(index), "tag_embedding_score")
    print("[raw] embedding scores written", flush=True)
    pool = pd.read_csv(RAW_SO / "stackoverflow_question_validation_pool.csv",
                       usecols=["question_id", "tag", "minutes_to_first_answer", "has_accepted_answer",
                                "answer_count", "is_closed"])
    _write(pool, "revealed_pool")
    backfill = pd.read_csv(next(RAW_SO.glob("*selenium_alias_backfill.csv")), usecols=["question_id", "creation_date"])
    _write(backfill, "selenium_backfill")
    collected = pd.read_csv(PROCESSED / "stackoverflow_2025_clean_question_tag.csv", usecols=["question_id"])
    counts = {**source_validation_counts(), "NCollectedTwentyFive": collected["question_id"].nunique(),
              "IndexWinStart": window[0].strftime("%Y-%m-%d"), "IndexWinEnd": window[1].strftime("%Y-%m-%d")}
    _write(pd.DataFrame({"key": list(counts), "value": list(counts.values())}), "collection_counts")
    print("[raw] analysis data complete", flush=True)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / f"{name}.csv.gz", float_precision="round_trip", **kw)


def _questions() -> pd.DataFrame:
    q = _read("question_tag", parse_dates=["day"], dtype={"control_mask": "int32", "curated": "int8"})
    q["creation_date"] = q["day"]
    q["week_start"] = q["day"] - pd.to_timedelta(q["day"].dt.weekday, unit="D")
    q["substitutable_type"] = (~q["question_type"].isin(NON_SUBSTITUTABLE)).astype(int)
    curated, score = q["curated"] == 1, q["score"].fillna(0)
    q["funnel"] = (curated & (score >= 0)).astype("int8")
    for k in range(5):
        q[f"s{k}"] = (curated & (score == k)).astype("int8")
    q["s5p"] = (curated & (score >= 5)).astype("int8")
    q["k_tags"] = q.groupby("question_id")["tag"].transform("size")
    return q


def load() -> Data:
    index = _read("tag_index")
    weeks = _read("weeks", parse_dates=["week_start"])
    counts = dict(zip(*_read("collection_counts", dtype=str)[["key", "value"]].T.values))
    scores = _read("question_type_substitutability")
    return Data(q=_questions(),
                main=finish_cells(_read("panel_main", parse_dates=["week_start"]), index),
                ext=finish_cells(_read("panel_extended", parse_dates=["week_start"]), index),
                index=index,
                weeks_main=pd.DatetimeIndex(weeks.loc[weeks["main_window"] == 1, "week_start"]),
                weeks_ext=pd.DatetimeIndex(weeks["week_start"]),
                dropped_weeks=_read("dropped_weeks", parse_dates=["week_start"]),
                index_window=(pd.Timestamp(counts["IndexWinStart"]), pd.Timestamp(counts["IndexWinEnd"])),
                counts=counts,
                sample=_read("validation_sample"),
                embed_tag=_read("tag_embedding_score", dtype={"embed": "float32"}),  # the precision of the model
                type_scores=dict(zip(scores["question_type"], scores["score"])),
                pool=_read("revealed_pool"),
                labels=_read(LABELS),
                backfill=_read("selenium_backfill"))
