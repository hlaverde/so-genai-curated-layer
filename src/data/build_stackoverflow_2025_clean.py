"""Build the clean 2025 question-tag dataset and its collection summary tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetch_stackoverflow_via_api import SELENIUM_ALIASES, week_start_of  # noqa: E402
from src.paths import LOGS_DIR, PROCESSED_DIR, RAW_DIR, TABLES_DIR  # noqa: E402

RAW_DIR_2025 = RAW_DIR / "stackoverflow" / "api_2025_full_100tags"
CLEAN_OUT = PROCESSED_DIR / "stackoverflow_2025_clean_question_tag.csv"
RAW_COVERAGE_OUT = TABLES_DIR / "coverage_2025_raw_files.csv"
TAG_COVERAGE_OUT = TABLES_DIR / "coverage_2025_by_tag.csv"
COUNTS_OUT = TABLES_DIR / "tag_counts_2024_2025.csv"
FAILED_WINDOWS = LOGS_DIR / "failed_2025_api_windows.csv"
MANIFEST_CANDIDATES = [
    RAW_DIR_2025 / "manifest_2025_full_100tags.csv",
    RAW_DIR_2025 / "_manifest_2025_full_100tags.csv",
]


def canonical_tag(value: str) -> str:
    return SELENIUM_ALIASES.get(value, value)


def load_fixed_tags(path: Path) -> pd.DataFrame:
    tags = pd.read_csv(path)
    if "tag" not in tags.columns:
        raise ValueError(f"{path} has no tag column")
    return tags


def manifest_path() -> Path | None:
    for path in MANIFEST_CANDIDATES:
        if path.exists():
            return path
    return None


def iter_raw_pages(raw_dir: Path):
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name.startswith("manifest"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("_metadata", {})
        yield path, payload, meta


def rows_from_payload(path: Path, payload: dict, meta: dict) -> list[dict]:
    out = []
    tag_consulted = canonical_tag(meta.get("tag_consulted") or meta.get("api_tag") or "")
    api_tag = meta.get("api_tag") or tag_consulted
    for q in payload.get("items", []):
        creation_ts = q.get("creation_date")
        if creation_ts is None:
            continue
        creation = pd.to_datetime(int(creation_ts), unit="s", utc=True)
        if creation.year != 2025:
            continue
        body = q.get("body") or ""
        owner = q.get("owner") or {}
        tags_original = q.get("tags") or []
        out.append({
            "question_id": q.get("question_id"),
            "tag": tag_consulted,
            "tag_consulted": tag_consulted,
            "api_tag": api_tag,
            "creation_date": creation.strftime("%Y-%m-%d %H:%M:%S"),
            "week": week_start_of(creation.to_pydatetime()),
            "week_start": week_start_of(creation.to_pydatetime()),
            "score": q.get("score", 0),
            "answer_count": q.get("answer_count", 0),
            "is_answered": q.get("is_answered"),
            "accepted_answer_id": q.get("accepted_answer_id"),
            "closed_date": q.get("closed_date"),
            "owner_user_id": owner.get("user_id"),
            "title": q.get("title", ""),
            "tags_original": "|".join(tags_original),
            "body_length": len(body),
            "has_code": 1 if "<code>" in body else 0,
            "has_accepted_answer": 1 if q.get("accepted_answer_id") else 0,
            "is_closed": 1 if q.get("closed_date") else 0,
            "source": "stackexchange_api_v2_3",
            "source_file": path.name,
            "fetch_timestamp": meta.get("fetch_timestamp"),
            "api_quota_remaining": meta.get("api_quota_remaining"),
            "api_page": meta.get("page"),
            "window_start": meta.get("window_start"),
            "window_end": meta.get("window_end"),
        })
    return out


def build_clean(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path, payload, meta in iter_raw_pages(raw_dir):
        rows.extend(rows_from_payload(path, payload, meta))
    if not rows:
        return pd.DataFrame(columns=[
            "question_id", "tag", "tag_consulted", "creation_date", "week",
            "score", "answer_count", "is_answered", "accepted_answer_id",
            "closed_date", "owner_user_id", "title", "tags_original",
            "body_length", "has_code", "source",
        ])
    df = pd.DataFrame(rows)
    df["question_id"] = pd.to_numeric(df["question_id"], errors="coerce")
    df = df.dropna(subset=["question_id", "tag_consulted"])
    df["question_id"] = df["question_id"].astype("int64")
    df["tag"] = df["tag"].map(canonical_tag)
    df["tag_consulted"] = df["tag_consulted"].map(canonical_tag)
    df = df.drop_duplicates(["question_id", "tag_consulted"]).reset_index(drop=True)
    df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["week"] = df["week_start"]
    for col in ["score", "answer_count", "body_length", "has_code", "has_accepted_answer", "is_closed", "owner_user_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_tag_coverage(clean: pd.DataFrame, fixed_tags: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    tags = fixed_tags[["tag"]].drop_duplicates().copy()
    files = []
    for path, payload, meta in iter_raw_pages(raw_dir):
        files.append({
            "tag": canonical_tag(meta.get("tag_consulted") or meta.get("api_tag") or ""),
            "api_tag": meta.get("api_tag"),
            "file": path.name,
            "n_items": len(payload.get("items", [])),
            "window_start": meta.get("window_start"),
            "window_end": meta.get("window_end"),
        })
    files_df = pd.DataFrame(files)
    mpath = manifest_path()
    manifest = pd.read_csv(mpath) if mpath is not None else pd.DataFrame()
    failed = pd.read_csv(FAILED_WINDOWS) if FAILED_WINDOWS.exists() else pd.DataFrame()

    expected_weeks = pd.date_range("2024-12-30", "2025-12-29", freq="W-MON")
    if clean.empty:
        by = pd.DataFrame(columns=["tag"])
    else:
        rows = []
        for tag, sub in clean.groupby("tag_consulted"):
            observed = set(pd.to_datetime(sub["week_start"]).dt.strftime("%Y-%m-%d"))
            missing = [w.strftime("%Y-%m-%d") for w in expected_weeks if w.strftime("%Y-%m-%d") not in observed]
            rows.append({
                "tag": tag,
                "n_question_tag_rows_2025": len(sub),
                "n_unique_questions_2025": sub["question_id"].nunique(),
                "n_weeks_observed_2025": sub["week_start"].nunique(),
                "first_observed_date": sub["creation_date"].min(),
                "last_observed_date": sub["creation_date"].max(),
                "missing_weeks": ";".join(missing),
                "duplicated_question_tag": int(sub.duplicated(["question_id", "tag_consulted"]).sum()),
            })
        by = pd.DataFrame(rows)

    by_tag = tags.merge(by, on="tag", how="left")
    if files_df.empty:
        by_tag["n_raw_json_files"] = 0
    else:
        fsum = files_df.groupby("tag").agg(n_raw_json_files=("file", "nunique")).reset_index()
        by_tag = by_tag.merge(fsum, on="tag", how="left")
    if manifest.empty:
        by_tag["n_api_calls_if_available"] = np.nan
        by_tag["max_window_days_used"] = np.nan
        by_tag["page_cap_hit"] = np.nan
    else:
        m = manifest.copy()
        m["tag"] = m["tag_consulted"].map(canonical_tag)
        m["window_start_dt"] = pd.to_datetime(m["window_start"], errors="coerce", utc=True)
        m["window_end_dt"] = pd.to_datetime(m["window_end"], errors="coerce", utc=True)
        m["window_days"] = (m["window_end_dt"] - m["window_start_dt"]).dt.days
        msum = m.groupby("tag").agg(
            n_api_calls_if_available=("api_calls", "sum"),
            max_window_days_used=("window_days", "max"),
            page_cap_hit=("page_cap_hit", "max"),
        ).reset_index()
        by_tag = by_tag.merge(msum, on="tag", how="left")
    if failed.empty:
        by_tag["failed_windows"] = 0
    else:
        failed["tag"] = failed["tag_consulted"].map(canonical_tag)
        fcnt = failed.groupby("tag").size().rename("failed_windows").reset_index()
        by_tag = by_tag.merge(fcnt, on="tag", how="left", suffixes=("", "_from_failed"))
        if "failed_windows_from_failed" in by_tag:
            by_tag["failed_windows"] = by_tag["failed_windows_from_failed"].fillna(by_tag["failed_windows"])
            by_tag = by_tag.drop(columns=["failed_windows_from_failed"])

    fill_zero = ["n_raw_json_files", "n_question_tag_rows_2025", "n_unique_questions_2025", "n_weeks_observed_2025", "failed_windows"]
    for col in fill_zero:
        if col not in by_tag.columns:
            by_tag[col] = 0
        by_tag[col] = pd.to_numeric(by_tag[col], errors="coerce").fillna(0).astype(int)
    if "page_cap_hit" not in by_tag.columns:
        by_tag["page_cap_hit"] = 0
    by_tag["page_cap_hit"] = pd.to_numeric(by_tag["page_cap_hit"], errors="coerce").fillna(0).astype(int)
    if "missing_weeks" not in by_tag.columns:
        by_tag["missing_weeks"] = ";".join(w.strftime("%Y-%m-%d") for w in expected_weeks)
    by_tag["suspicious_flag"] = ""
    by_tag.loc[by_tag["n_raw_json_files"] == 0, "suspicious_flag"] += "missing_raw;"
    by_tag.loc[by_tag["failed_windows"] > 0, "suspicious_flag"] += "failed_windows;"
    by_tag.loc[by_tag["page_cap_hit"] > 0, "suspicious_flag"] += "possible_page_cap;"
    by_tag.loc[by_tag["n_question_tag_rows_2025"] == 0, "suspicious_flag"] += "zero_2025;"
    by_tag.loc[by_tag["missing_weeks"].fillna("") != "", "suspicious_flag"] += "missing_observed_weeks;"
    first_dt = pd.to_datetime(by_tag["first_observed_date"], errors="coerce")
    last_dt = pd.to_datetime(by_tag["last_observed_date"], errors="coerce")
    by_tag.loc[first_dt > pd.Timestamp("2025-01-31"), "suspicious_flag"] += "late_first_observation;"
    by_tag.loc[last_dt < pd.Timestamp("2025-11-01"), "suspicious_flag"] += "early_last_observation;"
    by_tag["suspicious_flag"] = by_tag["suspicious_flag"].str.rstrip(";")
    by_tag["status"] = "ok"
    by_tag.loc[by_tag["n_raw_json_files"] == 0, "status"] = "missing_raw"
    by_tag.loc[by_tag["failed_windows"] > 0, "status"] = "failed_windows"
    by_tag.loc[by_tag["page_cap_hit"] > 0, "status"] = "page_cap_hit"
    by_tag.loc[first_dt > pd.Timestamp("2025-01-15"), "status"] = by_tag["status"] + ";late_first_observation"
    by_tag.loc[last_dt < pd.Timestamp("2025-11-01"), "status"] = by_tag["status"] + ";early_last_observation"
    keep = [
        "tag",
        "n_raw_json_files",
        "n_question_tag_rows_2025",
        "n_unique_questions_2025",
        "n_weeks_observed_2025",
        "first_observed_date",
        "last_observed_date",
        "missing_weeks",
        "failed_windows",
        "page_cap_hit",
        "suspicious_flag",
        "status",
    ]
    for col in keep:
        if col not in by_tag.columns:
            by_tag[col] = ""
    return by_tag[keep].sort_values("tag")


def build_raw_file_coverage(fixed_tags: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    tags = fixed_tags[["tag"]].drop_duplicates().copy()
    mpath = manifest_path()
    manifest = pd.read_csv(mpath) if mpath is not None else pd.DataFrame()
    failed = pd.read_csv(FAILED_WINDOWS) if FAILED_WINDOWS.exists() else pd.DataFrame()
    rows = []
    for tag in tags["tag"]:
        folder = raw_dir / tag
        files = sorted(folder.glob("*.json")) if folder.exists() else []
        status_bits = []
        manifest_status = ""
        page_cap = 0
        if not manifest.empty:
            m = manifest[manifest["tag_consulted"].map(canonical_tag) == tag]
            if not m.empty:
                manifest_status = ";".join(f"{k}:{v}" for k, v in m["status"].value_counts().sort_index().items())
                page_cap = int(pd.to_numeric(m.get("page_cap_hit"), errors="coerce").fillna(0).max())
        failed_n = 0
        if not failed.empty and "tag_consulted" in failed.columns:
            failed_n = int((failed["tag_consulted"].map(canonical_tag) == tag).sum())
        if not folder.exists() or not files:
            status_bits.append("missing_raw")
        if failed_n:
            status_bits.append("failed_windows")
        if page_cap:
            status_bits.append("possible_page_cap")
        if folder.exists() and len(files) < 1:
            status_bits.append("suspicious_low_files")
        if not status_bits:
            status_bits.append("ok_raw_present")
        rows.append({
            "tag": tag,
            "raw_folder_exists": bool(folder.exists()),
            "n_json_files": len(files),
            "first_json_file": files[0].name if files else "",
            "last_json_file": files[-1].name if files else "",
            "manifest_status_if_available": manifest_status,
            "failed_windows_if_available": failed_n,
            "page_cap_hit_if_available": page_cap,
            "status": ";".join(status_bits),
        })
    return pd.DataFrame(rows).sort_values("tag")


def build_counts(clean: pd.DataFrame, fixed_tags: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path, usecols=["tag", "week_start", "questions"])
    master["week_start"] = pd.to_datetime(master["week_start"])
    q24 = master[master["week_start"].dt.year == 2024].groupby("tag")["questions"].sum().rename("questions_2024")
    q25 = clean.groupby("tag_consulted")["question_id"].nunique().rename("questions_2025") if not clean.empty else pd.Series(dtype=float, name="questions_2025")
    out = fixed_tags.merge(q24, on="tag", how="left").merge(q25, left_on="tag", right_index=True, how="left")
    out["questions_2024"] = out["questions_2024"].fillna(0)
    out["questions_2025"] = out["questions_2025"].fillna(0)
    out["pct_change_2024_2025"] = np.where(out["questions_2024"] > 0, (out["questions_2025"] - out["questions_2024"]) / out["questions_2024"], np.nan)
    rename = {}
    if "ai_answerability_structural" in out.columns:
        rename["ai_answerability_structural"] = "ai_answerability_structural_if_available"
    if "embedding_answerability" in out.columns:
        rename["embedding_answerability"] = "embedding_answerability_if_available"
    out = out.rename(columns=rename)
    coverage = build_tag_coverage(clean, fixed_tags, RAW_DIR_2025)
    out = out.merge(
        coverage[["tag", "missing_weeks", "failed_windows", "page_cap_hit", "status"]],
        on="tag",
        how="left",
    )
    flags = []
    for _, row in out.iterrows():
        f = []
        if row["questions_2025"] == 0:
            f.append("zero_2025")
        if row["questions_2024"] >= 100 and row["questions_2025"] < 0.1 * row["questions_2024"]:
            f.append("implausibly_low_2025")
        if str(row.get("missing_weeks", "")) not in ["", "nan"]:
            f.append("missing_observed_weeks")
        if pd.to_numeric(row.get("failed_windows", 0), errors="coerce") > 0:
            f.append("failed_raw_windows")
        if pd.to_numeric(row.get("page_cap_hit", 0), errors="coerce") > 0:
            f.append("possible_page_cap")
        if "early_last_observation" in str(row.get("status", "")):
            f.append("early_last_observation")
        flags.append(";".join(f))
    out["suspicious_flag"] = flags
    keep = ["tag", "questions_2024", "questions_2025", "pct_change_2024_2025"]
    for col in ["ai_answerability_structural_if_available", "embedding_answerability_if_available"]:
        if col in out.columns:
            keep.append(col)
    keep.append("suspicious_flag")
    return out[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR_2025)
    parser.add_argument("--tag-source", type=Path, default=PROCESSED_DIR / "ai_answerability_index.csv")
    parser.add_argument("--master-2024", type=Path, default=PROCESSED_DIR / "stackoverflow_question_type_master_panel.csv")
    parser.add_argument("--clean-output", type=Path, default=CLEAN_OUT)
    args = parser.parse_args()

    fixed = load_fixed_tags(args.tag_source)
    clean = build_clean(args.raw_dir)
    args.clean_output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.clean_output, index=False)
    print(f"clean rows={len(clean):,}, tags={clean['tag_consulted'].nunique() if not clean.empty else 0}")
    print(f"saved {args.clean_output}")
    # coverage tables of the collection (not inputs to the analysis)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    build_raw_file_coverage(fixed, args.raw_dir).to_csv(RAW_COVERAGE_OUT, index=False)
    build_tag_coverage(clean, fixed, args.raw_dir).to_csv(TAG_COVERAGE_OUT, index=False)
    print(f"saved {RAW_COVERAGE_OUT}")
    print(f"saved {TAG_COVERAGE_OUT}")
    if args.master_2024.exists():
        build_counts(clean, fixed, args.master_2024).to_csv(COUNTS_OUT, index=False)
        print(f"saved {COUNTS_OUT}")


if __name__ == "__main__":
    main()
