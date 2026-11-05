"""Prepare question-type mechanism panels from lightweight SEDE exports."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import LOGS_DIR, PROCESSED_DIR, RAW_DIR
from src.utils.io import write_csv
from src.utils.logging_utils import get_logger


RAW_SUBDIR = RAW_DIR / "stackoverflow"
OUTPUT_FILE = PROCESSED_DIR / "stackoverflow_question_type_week_panel.csv"

TAG_ALIASES = {
    "selenium-webdriver": "selenium",
    "selenium-chromedriver": "selenium",
    "webdriver": "selenium",
    "chromedriver": "selenium",
}

# Title keywords of each question type. The order of RULES is the precedence of
# the classification: a title takes the first type whose keywords it contains.
VERSION_ENV_WORDS = ("version", "environment", "windows", "linux", "mac", "ubuntu", "install", "installed",
                     "configuration", "dependency", "package", "build error", "compiler", "runtime")
ADVANCED_WORDS = ("architecture", "design pattern", "scalable", "performance", "optimization",
                  "microservice", "distributed", "best practice")
DEBUG_WORDS = ("error", "exception", "traceback", "not working", "bug", "debug", "failed", "failure",
               "crash", "segmentation fault")
HOWTO_WORDS = ("how to", "how do i", "how can i", "what is the way", "how should i")


def words_pattern(words: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", flags=re.IGNORECASE)


VERSION_ENV_PAT = words_pattern(VERSION_ENV_WORDS)
ADVANCED_PAT = words_pattern(ADVANCED_WORDS)
DEBUG_PAT = words_pattern(DEBUG_WORDS)
HOWTO_PAT = words_pattern(HOWTO_WORDS)
RULES = [("version_environment_specific", VERSION_ENV_PAT), ("advanced_architecture", ADVANCED_PAT),
         ("debugging_simple", DEBUG_PAT), ("how_to", HOWTO_PAT)]
SHORT_BODY_MAX = 1200            # characters of the body, HTML included
NON_SUBSTITUTABLE = ("version_environment_specific", "advanced_architecture")
CONTROL_WORDS = VERSION_ENV_WORDS + ADVANCED_WORDS
# zero-width space, non-joiner and joiner, word joiner, zero-width no-break space
ZERO_WIDTH_PAT = re.compile("[​‌‍⁠﻿]")


def source_files(raw_dir: Path, year: int | None) -> list[Path]:
    if year is None:
        files = sorted(raw_dir.glob("stackoverflow_question_type_raw_*.csv"))
    elif year == 2020:
        files = []
        files += sorted(raw_dir.glob("stackoverflow_question_type_raw_2020_01_w*.csv"))
        files += sorted(raw_dir.glob("stackoverflow_question_type_raw_2020_01_p5.csv"))
        files += sorted(raw_dir.glob("stackoverflow_question_type_raw_2020-*.csv"))
    else:
        files = sorted(raw_dir.glob(f"stackoverflow_question_type_raw_{year}-*.csv"))
    return [
        path
        for path in files
        if path.name != "stackoverflow_question_type_raw_2020_01.csv"
    ]


def clean_titles(title: pd.Series) -> pd.Series:
    return title.fillna("").str.replace(ZERO_WIDTH_PAT, "", regex=True)


def classify_questions(df: pd.DataFrame) -> pd.DataFrame:
    title = clean_titles(df["title"])
    body_length = pd.to_numeric(df["body_length"], errors="coerce").fillna(0)
    has_code = pd.to_numeric(df["has_code"], errors="coerce").fillna(0).astype(int)

    matches = {name: title.str.contains(pat, na=False) for name, pat in RULES}
    short_code = (has_code == 1) & (body_length <= SHORT_BODY_MAX)
    long_code = (has_code == 1) & (body_length > SHORT_BODY_MAX)

    conditions = [*matches.values(), short_code, long_code]
    choices = [*matches, "short_code", "long_code"]
    df = df.copy()
    df["question_type"] = np.select(conditions, choices, default="other_conceptual")
    df["substitutable_type"] = np.where(matches[NON_SUBSTITUTABLE[0]] | matches[NON_SUBSTITUTABLE[1]], 0, 1)
    return df


def control_word_mask(title: pd.Series) -> np.ndarray:
    """Bit j set when the title contains CONTROL_WORDS[j]; zero for a substitutable title."""
    title = clean_titles(title)
    mask = np.zeros(len(title), dtype=np.int32)
    hit = title.str.contains(words_pattern(CONTROL_WORDS), na=False).to_numpy()
    sub = title[hit]
    bits = np.zeros(len(sub), dtype=np.int32)
    for j, word in enumerate(CONTROL_WORDS):
        bits |= sub.str.contains(words_pattern((word,)), na=False).to_numpy(dtype=np.int32) << j
    mask[hit] = bits
    return mask


def load_raw(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No question-type raw files found.")
    df = pd.concat(frames, ignore_index=True)
    df["tag"] = df["tag"].replace(TAG_ALIASES)
    df = df.drop_duplicates(["question_id", "tag"]).reset_index(drop=True)
    df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    numeric_cols = [
        "question_id",
        "owner_user_id",
        "body_length",
        "has_code",
        "score",
        "answer_count",
        "has_accepted_answer",
        "is_closed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def check_raw(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(df),
        "date_min": df["creation_date"].min(),
        "date_max": df["creation_date"].max(),
        "tags": df["tag"].nunique(),
        "duplicate_question_tag": int(df.duplicated(["question_id", "tag"]).sum()),
        "source_files": df["source_file"].nunique(),
    }


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = classify_questions(df)
    grouped = (
        df.groupby(["tag", "week_start", "question_type", "substitutable_type"], dropna=False)
        .agg(
            questions=("question_id", "nunique"),
            answers=("answer_count", "sum"),
            accepted_answers=("has_accepted_answer", "sum"),
            avg_score=("score", "mean"),
            closed_questions=("is_closed", "sum"),
            unique_users=("owner_user_id", pd.Series.nunique),
            body_length_mean=("body_length", "mean"),
            code_questions=("has_code", "sum"),
        )
        .reset_index()
    )
    grouped["answer_rate"] = np.where(
        grouped["questions"] > 0, grouped["answers"] / grouped["questions"], np.nan
    )
    grouped["accepted_share"] = np.where(
        grouped["questions"] > 0, grouped["accepted_answers"] / grouped["questions"], np.nan
    )
    grouped["closed_share"] = np.where(
        grouped["questions"] > 0, grouped["closed_questions"] / grouped["questions"], np.nan
    )
    grouped["code_share"] = np.where(
        grouped["questions"] > 0, grouped["code_questions"] / grouped["questions"], np.nan
    )
    return grouped.sort_values(["tag", "week_start", "question_type"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=RAW_SUBDIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    logger = get_logger(
        "prepare_stackoverflow_question_type_raw",
        LOGS_DIR / "prepare_stackoverflow_question_type_raw.log",
    )
    files = source_files(args.raw_dir, args.year)
    logger.info("Loading %s raw files", len(files))
    raw = load_raw(files)
    checks = check_raw(raw)
    logger.info("Raw checks: %s", checks)
    if checks["duplicate_question_tag"]:
        raise ValueError(f"Duplicate question-tag rows detected: {checks['duplicate_question_tag']}")
    panel = build_panel(raw)
    write_csv(panel, args.output)
    logger.info("Wrote panel with %s rows to %s", len(panel), args.output)


if __name__ == "__main__":
    main()
