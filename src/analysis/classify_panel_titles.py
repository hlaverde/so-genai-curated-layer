"""Nearest-centroid labels of all-MiniLM-L6-v2, with the exemplar questions of
question_type_classifiers, for the title of every question in the main panel.
Derived input of validation.minilm_panel. Titles are unescaped
(API titles come HTML-escaped) and stripped of zero-width characters. Finished
chunks stay on disk, so an interrupted run resumes where it stopped."""
from __future__ import annotations

import argparse
import html
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pandas as pd

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK = 5000
_state: dict = {}


def _init() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from sentence_transformers import SentenceTransformer
    from src.analysis.question_type_classifiers import seed_centroids
    torch.set_num_threads(1)
    model = SentenceTransformer(MODEL, device="cpu")
    _state["model"], (_state["names"], _state["centroids"]) = model, seed_centroids(model)


def _label(job: tuple[str, list[str]]) -> None:
    path, titles = job
    emb = _state["model"].encode(titles, batch_size=32, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
    with open(path + ".part", "wb") as f:
        np.save(f, (emb @ _state["centroids"].T).argmax(axis=1).astype(np.int8))
    os.replace(path + ".part", path)


def main() -> None:
    from src.analysis.question_type_classifiers import SEEDS
    from src.data.prepare_stackoverflow_question_type_raw import clean_titles
    from src.pipeline.common import ANALYSIS, PROCESSED
    from src.pipeline.data import complete_weeks, load_questions

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default=str(ANALYSIS / "minilm_panel_title_labels.csv.gz"))
    args = parser.parse_args()
    q = load_questions(keep_title=True)
    main_weeks = complete_weeks(q)[0]
    q = q.loc[q["week_start"].isin(main_weeks), ["question_id", "title"]].drop_duplicates("question_id")
    q["clean"] = clean_titles(q["title"]).map(html.unescape)
    titles = np.sort(q["clean"].unique())
    print(f"{len(q):,} questions, {len(titles):,} distinct titles", flush=True)
    chunk_dir = PROCESSED / "_minilm_chunks"
    chunk_dir.mkdir(exist_ok=True)
    jobs = [(str(chunk_dir / f"chunk_{i:05d}.npy"), titles[i * CHUNK:(i + 1) * CHUNK].tolist())
            for i in range(int(np.ceil(len(titles) / CHUNK)))]
    pending = [j for j in jobs if not os.path.exists(j[0])]
    t0, done = time.time(), len(jobs) - len(pending)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for k, _ in enumerate(pool.imap_unordered(_label, pending), 1):
            if k % 20 == 0:
                print(f"{done + k}/{len(jobs)} chunks, {k * CHUNK / (time.time() - t0):.0f} titles/s", flush=True)
    labels = np.concatenate([np.load(j[0]) for j in jobs])
    names = list(SEEDS)
    q["minilm_type"] = pd.Series(labels, index=titles).reindex(q["clean"]).map(lambda i: names[i]).to_numpy()
    q[["question_id", "minilm_type"]].sort_values("question_id").to_csv(args.output, index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    shutil.rmtree(chunk_dir)
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
