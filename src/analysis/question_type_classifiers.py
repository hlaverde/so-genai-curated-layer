"""Question-type classifiers used in the validation: the title/body heuristic,
two sentence-embedding classifiers (all-MiniLM-L6-v2 and all-mpnet-base-v2),
the stratified validation sample, and Fleiss' kappa."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.prepare_stackoverflow_question_type_raw import (
    ADVANCED_PAT, DEBUG_PAT, HOWTO_PAT, SHORT_BODY_MAX, VERSION_ENV_PAT, ZERO_WIDTH_PAT)


CHATGPT_RELEASE = pd.Timestamp("2022-11-30")
SEED = 20251122

SUBSTITUTABLE = {
    "short_code", "long_code", "how_to", "debugging_simple",
    "other_conceptual",
}

SEEDS = {
    "short_code": [
        "Why does this regex not match digits in python?",
        "How to remove duplicates from a list in JavaScript?",
        "Sorting a Pandas dataframe by column value",
        "How to read a csv into Pandas",
        "Python list comprehension filter elements",
        "JavaScript split string by comma",
        "Convert dictionary to JSON string Python",
        "Get current date in JavaScript",
        "How to reverse a string in Java",
        "Format a number with two decimals",
        "Iterate through a dictionary in Python",
        "Convert list to numpy array",
    ],
    "long_code": [
        "TypeError when running a flask app with multiple endpoints",
        "Spring Boot REST API returns 404 for resource",
        "Reactjs state hook does not update after async fetch",
        "Django model migration fails with foreign key constraint",
        "Node.js Express middleware order causes wrong route handler",
        "Java Stream collect to map ignores duplicate keys",
        "Angular HTTP interceptor not catching errors",
        "Kubernetes pod fails to mount persistent volume",
        "Tensorflow training loop OOM on GPU memory",
        "Spring Security OAuth2 redirect URI mismatch",
        "Vue.js Vuex action not committing mutation in production",
        "Rails ActiveRecord query returns wrong subset under join",
    ],
    "how_to": [
        "How to deploy a Django app on AWS Elastic Beanstalk?",
        "How do I authenticate users with Firebase in React?",
        "How can I configure CORS in Express?",
        "How to use environment variables in Docker container?",
        "What is the way to schedule background jobs in Python?",
        "How should I structure folders in a Vue.js project?",
        "How do I migrate a SQL Server database to Postgres?",
        "How to handle JWT refresh tokens in Angular?",
        "How can I integrate Stripe with my Flask app?",
        "How to enable HTTPS for an nginx reverse proxy?",
        "How can I implement rate limiting in Node.js?",
        "How do I set up a CI pipeline in GitHub Actions?",
    ],
    "debugging_simple": [
        "Error: connection refused on port 8080",
        "Unhandled promise rejection in async function",
        "Bug: button onClick fires twice in React",
        "SegFault when running C++ program on Linux",
        "NullPointerException at startup in Spring Boot",
        "TypeError cannot read property of undefined in JS",
        "RuntimeError: tensor sizes mismatch in PyTorch",
        "Crash on Android app launch with Firebase init",
        "Build failed compile error in Visual Studio 2022",
        "Failed to install package via npm: missing dependency",
        "Exception thrown when calling REST API in Java client",
        "Traceback: KeyError when processing CSV row",
    ],
    "other_conceptual": [
        "Difference between abstract class and interface in Java",
        "When should I use Redis vs Memcached?",
        "Why is async/await preferred over callbacks?",
        "What does the V in MVC actually do?",
        "When does immutability matter in functional programming?",
        "Difference between unit, integration and end-to-end tests",
        "Why use composition over inheritance?",
        "What is the purpose of dependency injection?",
        "Trade-offs between REST and GraphQL APIs",
        "Why are pure functions important in functional code?",
        "What is referential transparency and why does it matter?",
        "Difference between mutex and semaphore concurrency primitives",
    ],
    "version_environment_specific": [
        "Python 3.11 incompatible with package X version Y",
        "Configuration of Windows path for Node v18 installation",
        "Ubuntu 20.04 dependency conflict during apt install",
        "macOS Ventura broke my Homebrew Python environment",
        "Docker compose version 3 deprecation warning",
        "Java 17 module path runtime error",
        "Build error Visual Studio 2019 Windows SDK 10 missing",
        "Anaconda environment activation fails in Linux WSL",
        "Compiler runtime mismatch between gcc 11 and 12",
        "Package npm install fails on M1 Mac native arm64",
        "Cuda toolkit version incompatible with installed driver",
        "Setup configuration file missing for vcpkg on Windows 11",
    ],
    "advanced_architecture": [
        "Best practice for designing microservice boundaries",
        "Scalable distributed message-queue architecture for events",
        "Performance optimization of read-heavy database with sharding",
        "Design pattern for event-driven communication between services",
        "Architectural choice CQRS vs traditional CRUD for high writes",
        "How to design fault-tolerant pipeline in Apache Kafka",
        "Best practice REST API versioning at scale",
        "Distributed cache strategy for multi-region system",
        "Optimization of GraphQL resolver performance",
        "Designing a polyglot persistence layer with multiple databases",
        "Best practice for event sourcing in domain-driven design",
        "Scalable strategy for distributed transactions across services",
    ],
}


def classify_heuristic(row) -> str:
    title = ZERO_WIDTH_PAT.sub("", str(row["title"])) if pd.notna(row["title"]) else ""
    body_length = int(row["body_length"]) if pd.notna(row["body_length"]) else 0
    has_code = int(row["has_code"]) if pd.notna(row["has_code"]) else 0
    if VERSION_ENV_PAT.search(title):
        return "version_environment_specific"
    if ADVANCED_PAT.search(title):
        return "advanced_architecture"
    if DEBUG_PAT.search(title):
        return "debugging_simple"
    if HOWTO_PAT.search(title):
        return "how_to"
    if has_code == 1 and body_length <= SHORT_BODY_MAX:
        return "short_code"
    if has_code == 1 and body_length > SHORT_BODY_MAX:
        return "long_code"
    return "other_conceptual"


def load_stratified_sample(raw_dir: Path, n: int, seed: int) -> pd.DataFrame:
    files = sorted(raw_dir.glob("stackoverflow_question_type_raw_2022-*.csv")) + \
            sorted(raw_dir.glob("stackoverflow_question_type_raw_2023-*.csv")) + \
            sorted(raw_dir.glob("stackoverflow_question_type_raw_2024-*.csv"))
    frames = []
    for f in files[:60]:
        try:
            frames.append(pd.read_csv(
                f, usecols=["question_id", "title", "body_length", "has_code"]
            ))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("question_id").reset_index(drop=True)
    df = df[df["title"].notna() & (df["title"].str.len() > 5)].reset_index(drop=True)
    df["heuristic_type"] = df.apply(classify_heuristic, axis=1)

    rng = np.random.default_rng(seed)
    per_cat = n // len(SEEDS)
    samples = []
    for cat in SEEDS.keys():
        sub = df[df["heuristic_type"] == cat]
        if len(sub) == 0:
            continue
        k = min(per_cat, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        samples.append(sub.iloc[idx])
    return pd.concat(samples, ignore_index=True)


def seed_centroids(model) -> tuple[list[str], np.ndarray]:
    """Category names and the unit-norm centroids of their exemplar questions."""
    cat_names = []
    centroids = []
    for cat, seeds in SEEDS.items():
        emb_seeds = model.encode(
            seeds, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        centroids.append(emb_seeds.mean(axis=0))
        cat_names.append(cat)
    centroids = np.stack(centroids)
    return cat_names, centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def embed_and_assign(titles, model_name) -> list[str]:
    from sentence_transformers import SentenceTransformer
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(titles)} titles...")
    emb_sample = model.encode(
        titles, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    print("Embedding seed centroids...")
    cat_names, centroids = seed_centroids(model)
    sims = emb_sample @ centroids.T
    assignments = sims.argmax(axis=1)
    return [cat_names[i] for i in assignments]


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for multi-rater categorical agreement.

    Args:
        ratings: matrix (n_items, n_categories) where each row sums to
            n_raters and counts how many raters assigned each category.
    Returns:
        Fleiss kappa.
    """
    n, k = ratings.shape
    raters = ratings.sum(axis=1)[0]
    p_j = ratings.sum(axis=0) / (n * raters)
    P_i = (
        (ratings * (ratings - 1)).sum(axis=1) / (raters * (raters - 1))
    )
    Pbar = P_i.mean()
    PE = (p_j ** 2).sum()
    return float((Pbar - PE) / (1 - PE)) if (1 - PE) > 0 else float("nan")
