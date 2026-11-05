"""Embedding-based substitutability score of the question types."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_substitutability_score_from_embeddings(
    panel: pd.DataFrame, validation_sample: pd.DataFrame, model_name: str
) -> dict[str, float]:
    """Continuous substitutability score of each question type: mean cosine
    similarity of its embedding centroid to the centroids of the
    substitutable types minus that to the non-substitutable types.
    Returns a dict question_type -> score."""
    from sentence_transformers import SentenceTransformer

    SEEDS = {
        "short_code": validation_sample[
            validation_sample["heuristic_type"] == "short_code"
        ]["title"].sample(min(50, (validation_sample["heuristic_type"] == "short_code").sum()), random_state=0).tolist(),
        "long_code": validation_sample[
            validation_sample["heuristic_type"] == "long_code"
        ]["title"].sample(min(50, (validation_sample["heuristic_type"] == "long_code").sum()), random_state=0).tolist(),
        "how_to": validation_sample[
            validation_sample["heuristic_type"] == "how_to"
        ]["title"].sample(min(50, (validation_sample["heuristic_type"] == "how_to").sum()), random_state=0).tolist(),
        "debugging_simple": validation_sample[
            validation_sample["heuristic_type"] == "debugging_simple"
        ]["title"].sample(min(50, (validation_sample["heuristic_type"] == "debugging_simple").sum()), random_state=0).tolist(),
        "other_conceptual": validation_sample[
            validation_sample["heuristic_type"] == "other_conceptual"
        ]["title"].sample(min(50, (validation_sample["heuristic_type"] == "other_conceptual").sum()), random_state=0).tolist(),
        "version_environment_specific": validation_sample[
            validation_sample["heuristic_type"] == "version_environment_specific"
        ]["title"].sample(min(50, (validation_sample["heuristic_type"] == "version_environment_specific").sum()), random_state=0).tolist(),
        "advanced_architecture": validation_sample[
            validation_sample["heuristic_type"] == "advanced_architecture"
        ]["title"].sample(min(50, (validation_sample["heuristic_type"] == "advanced_architecture").sum()), random_state=0).tolist(),
    }

    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)

    # centroid of each category
    centroids = {}
    for cat, texts in SEEDS.items():
        if len(texts) == 0:
            centroids[cat] = None
            continue
        emb = model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        centroids[cat] = emb.mean(axis=0)

    # score of category k: mean cosine similarity to the substitutable
    # categories minus mean cosine similarity to the non-substitutable ones
    SUB = {"short_code", "long_code", "how_to", "debugging_simple", "other_conceptual"}
    NONSUB = {"version_environment_specific", "advanced_architecture"}

    scores = {}
    for cat, centroid in centroids.items():
        if centroid is None:
            scores[cat] = 0.0
            continue
        sim_sub = np.mean([
            float(centroid @ centroids[s] / (np.linalg.norm(centroid) * np.linalg.norm(centroids[s])))
            for s in SUB if centroids.get(s) is not None and s != cat
        ])
        sim_nonsub = np.mean([
            float(centroid @ centroids[ns] / (np.linalg.norm(centroid) * np.linalg.norm(centroids[ns])))
            for ns in NONSUB if centroids.get(ns) is not None and ns != cat
        ])
        scores[cat] = sim_sub - sim_nonsub

    return scores
