import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import CHATGPT_RELEASE_DATE
from src.paths import PROCESSED_DIR, RAW_DIR, ensure_directories
from src.utils.io import read_csv, write_csv
from src.utils.logging_utils import get_logger
from src.utils.validation import require_columns


COMPLEXITY_PATTERN = "stackoverflow_post_complexity_tag_week_*.csv"
USER_GROUP_PATTERN = "stackoverflow_user_group_tag_week_20??.csv"


COMPONENTS = [
    "accepted_answer_rate_pre",
    "answer_rate_pre",
    "historical_frequency_pre",
    "tag_maturity_weeks_pre",
    "short_code_share_pre",
    "how_to_share_pre",
]
STRUCTURAL_COMPONENTS = [
    "accepted_answer_rate_pre",
    "answer_rate_pre",
    "historical_frequency_pre",
    "tag_maturity_weeks_pre",
]


def _zavg(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    zcols = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        std = values.std(ddof=0)
        zcols.append(pd.Series(0.0, index=df.index) if std == 0 else (values - values.mean()) / std)
    return pd.concat(zcols, axis=1).mean(axis=1)


def _pca(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    if len(values) < 2:
        return pd.Series(np.zeros(len(values)), index=df.index)
    scaled = StandardScaler().fit_transform(values)
    return pd.Series(PCA(n_components=1).fit_transform(scaled).ravel(), index=df.index)


def _quantile(series: pd.Series) -> pd.Series:
    bins = min(5, series.nunique())
    if bins <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return pd.qcut(series.rank(method="average"), q=bins, labels=False, duplicates="drop") / (bins - 1)


def load_complexity(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob(COMPLEXITY_PATTERN))
    paths = [p for p in paths if "2021_2023" in p.name or "2018_2020" in p.name or "2024_2026" in p.name]
    if not paths:
        raise FileNotFoundError(f"No complexity CSVs found in {input_dir}")
    frames = []
    for path in paths:
        frame = read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["week_start"] = pd.to_datetime(out["week_start"])
    out = out.drop_duplicates(["tag", "week_start"])
    return out.sort_values(["tag", "week_start"])


def load_user_groups(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob(USER_GROUP_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No user-group CSVs found in {input_dir}")
    frames = []
    for path in paths:
        frame = read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["week_start"] = pd.to_datetime(out["week_start"])
    out = out.drop_duplicates(["tag", "week_start", "new_user", "low_reputation_user"])
    return out.sort_values(["tag", "week_start", "new_user", "low_reputation_user"])


def build_ai_answerability_from_aggregates(tag_week: pd.DataFrame, complexity: pd.DataFrame) -> pd.DataFrame:
    require_columns(tag_week, ["tag", "week_start", "questions", "accepted_answers", "answer_rate"], "tag_week")
    require_columns(complexity, ["tag", "week_start", "short_code_share", "how_to_share"], "complexity")
    tw = tag_week.copy()
    tw["week_start"] = pd.to_datetime(tw["week_start"])
    pre_tw = tw[tw["week_start"] < pd.Timestamp(CHATGPT_RELEASE_DATE)].copy()
    pre_tw["active_week"] = pre_tw["questions"] > 0
    tag_stats = (
        pre_tw.groupby("tag", as_index=False)
        .agg(
            questions_pre=("questions", "sum"),
            accepted_answers_pre=("accepted_answers", "sum"),
            answer_rate_pre=("answer_rate", "mean"),
            historical_frequency_pre=("questions", "sum"),
            tag_maturity_weeks_pre=("active_week", "sum"),
        )
    )
    tag_stats["accepted_answer_rate_pre"] = (
        tag_stats["accepted_answers_pre"] / tag_stats["questions_pre"].replace({0: np.nan})
    ).fillna(0)

    comp = complexity.copy()
    pre_comp = comp[comp["week_start"] < pd.Timestamp(CHATGPT_RELEASE_DATE)]
    comp_stats = (
        pre_comp.groupby("tag", as_index=False)
        .agg(
            short_code_share_pre=("short_code_share", "mean"),
            how_to_share_pre=("how_to_share", "mean"),
        )
    )
    out = tag_stats.merge(comp_stats, on="tag", how="left").fillna(0)
    out["ai_answerability_zscore"] = _zavg(out, COMPONENTS)
    out["ai_answerability_pca"] = _pca(out, COMPONENTS)
    out["ai_answerability_quantile"] = _quantile(out["ai_answerability_zscore"])
    out["ai_answerability_structural"] = _zavg(out, STRUCTURAL_COMPONENTS)
    return out.sort_values("tag")


def prepare_answerability_inputs(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    tag_week = read_csv(input_dir / "stackoverflow_tag_week.csv")
    tag_week["week_start"] = pd.to_datetime(tag_week["week_start"])
    tag_week["post_chatgpt"] = tag_week["week_start"] >= pd.Timestamp(CHATGPT_RELEASE_DATE)
    complexity = load_complexity(input_dir)
    complexity["post_chatgpt"] = complexity["week_start"] >= pd.Timestamp(CHATGPT_RELEASE_DATE)
    try:
        user_groups = load_user_groups(input_dir)       # optional; not used by the answerability index
        user_groups["post_chatgpt"] = user_groups["week_start"] >= pd.Timestamp(CHATGPT_RELEASE_DATE)
    except FileNotFoundError:
        user_groups = None
    ai = build_ai_answerability_from_aggregates(tag_week, complexity)
    panel = tag_week.merge(
        ai[
            [
                "tag",
                "ai_answerability_zscore",
                "ai_answerability_pca",
                "ai_answerability_quantile",
                "ai_answerability_structural",
            ]
        ],
        on="tag",
        how="left",
    ).merge(complexity, on=["tag", "week_start", "post_chatgpt"], how="left", suffixes=("", "_complexity"))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tag_week_panel": output_dir / "stackoverflow_tag_week_panel.csv",
        "complexity": output_dir / "stackoverflow_complexity_tag_week.csv",
        "user_groups": output_dir / "stackoverflow_user_group_tag_week.csv",
        "ai_answerability": output_dir / "ai_answerability_index.csv",
    }
    write_csv(panel, outputs["tag_week_panel"])
    write_csv(complexity, outputs["complexity"])
    if user_groups is not None:
        write_csv(user_groups, outputs["user_groups"])
    write_csv(ai, outputs["ai_answerability"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the tag-level answerability index from the SEDE extracts.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR / "stackoverflow")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    logger = get_logger(__name__)
    outputs = prepare_answerability_inputs(args.input_dir, args.output_dir)
    for key, path in outputs.items():
        logger.info("Wrote %s to %s", key, path)


if __name__ == "__main__":
    main()
