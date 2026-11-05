"""Fetch calendar-year 2025 Stack Overflow questions for the fixed 100-tag sample.

This is a resumable Stack Exchange API v2.3 collector. It keeps the fixed
pre-treatment 100 tags from `ai_answerability_index.csv`, applies the Selenium
alias rule, and writes one raw JSON file per successful API page.

The script requires `STACKEXCHANGE_KEY` for the full run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetch_stackoverflow_via_api import (  # noqa: E402
    API_BASE,
    DEFAULT_FILTER_ID,
    SELENIUM_ALIASES,
    SITE,
    TOP_100_TAGS,
    create_filter,
)
from src.paths import LOGS_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

START_2025 = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_2025 = datetime(2026, 1, 1, tzinfo=timezone.utc)
RAW_OUT_DIR = RAW_DIR / "stackoverflow" / "api_2025_full_100tags"
MANIFEST = RAW_OUT_DIR / "manifest_2025_full_100tags.csv"
FAILED_WINDOWS = LOGS_DIR / "failed_2025_api_windows.csv"
SOURCE = "stackexchange_api_v2_3"
PAGE_SIZE = 100
LOW_QUOTA_STOP = 50
MAX_PAGES_WITH_KEY = 199
MAX_PAGES_NO_KEY = 24


@dataclass(frozen=True)
class Window:
    tag_consulted: str
    api_tag: str
    start: datetime
    end: datetime
    level: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.api_tag, self.start.isoformat(), self.end.isoformat())


def ts(dt: datetime) -> int:
    return int(dt.timestamp())


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def load_fixed_tags(path: Path) -> list[str]:
    if path.exists():
        df = pd.read_csv(path)
        if "tag" not in df.columns:
            raise ValueError(f"{path} has no 'tag' column")
        tags = sorted(df["tag"].dropna().astype(str).unique().tolist())
    else:
        tags = sorted(TOP_100_TAGS)
    if len(tags) != 100:
        raise ValueError(f"Expected 100 fixed tags, found {len(tags)} in {path}")
    return tags


def query_targets(tags: Iterable[str]) -> list[tuple[str, str]]:
    targets = [(tag, tag) for tag in tags]
    if "selenium" in set(tags):
        targets.extend(("selenium", alias) for alias in sorted(SELENIUM_ALIASES))
    return targets


def completed_keys(manifest_path: Path) -> set[tuple[str, str, str]]:
    if not manifest_path.exists():
        return set()
    out: set[tuple[str, str, str]] = set()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete":
                out.add((row["api_tag"], row["window_start"], row["window_end"]))
    return out


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def split_window(win: Window) -> list[Window]:
    span_days = (win.end - win.start).days
    if span_days > 180:
        level = "semiannual"
    elif span_days > 31:
        level = "monthly"
    elif span_days > 7:
        level = "weekly"
    else:
        level = "daily"

    if level == "semiannual":
        dates = [datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 7, 1, tzinfo=timezone.utc), END_2025]
    elif level == "monthly":
        dates = [win.start]
        cur = datetime(win.start.year, win.start.month, 1, tzinfo=timezone.utc)
        while True:
            nxt = datetime(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1, tzinfo=timezone.utc)
            if nxt >= win.end:
                break
            dates.append(nxt)
            cur = nxt
        dates.append(win.end)
    elif level == "weekly":
        dates = [win.start]
        cur = win.start
        while cur < win.end:
            cur = min(cur + pd.Timedelta(days=7).to_pytimedelta(), win.end)
            dates.append(cur)
    else:
        dates = [win.start]
        cur = win.start
        while cur < win.end:
            cur = min(cur + pd.Timedelta(days=1).to_pytimedelta(), win.end)
            dates.append(cur)

    children = []
    for a, b in zip(dates[:-1], dates[1:]):
        if a < b:
            children.append(Window(win.tag_consulted, win.api_tag, a, b, level))
    return children


def initial_windows(tag_consulted: str, api_tag: str, level: str) -> list[Window]:
    if level == "annual":
        return [Window(tag_consulted, api_tag, START_2025, END_2025, "annual")]
    if level not in {"semiannual", "monthly", "weekly", "daily"}:
        raise ValueError(f"Unsupported initial level: {level}")

    if level == "semiannual":
        boundaries = [
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 7, 1, tzinfo=timezone.utc),
            END_2025,
        ]
    elif level == "monthly":
        boundaries = [datetime(2025, month, 1, tzinfo=timezone.utc) for month in range(1, 13)]
        boundaries.append(END_2025)
    elif level == "weekly":
        boundaries = [START_2025]
        cur = START_2025
        while cur < END_2025:
            cur = min(cur + pd.Timedelta(days=7).to_pytimedelta(), END_2025)
            boundaries.append(cur)
    else:
        boundaries = [START_2025]
        cur = START_2025
        while cur < END_2025:
            cur = min(cur + pd.Timedelta(days=1).to_pytimedelta(), END_2025)
            boundaries.append(cur)

    return [
        Window(tag_consulted, api_tag, start, end, level)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]


def raw_page_path(win: Window, page: int) -> Path:
    safe_tag = win.api_tag.replace("#", "sharp").replace("+", "plus").replace(".", "dot")
    name = f"{safe_tag}_{iso(win.start)}_{iso(win.end)}_page{page:03d}.json"
    return RAW_OUT_DIR / win.tag_consulted / name


def fetch_leaf(session: requests.Session, win: Window, key: str | None, filter_id: str, max_pages: int, polite_sleep: float) -> dict:
    page = 1
    calls = 0
    items = 0
    last_quota = None
    page_cap_hit = False
    while True:
        params = {
            "site": SITE,
            "tagged": win.api_tag,
            "fromdate": ts(win.start),
            "todate": ts(win.end),
            "pagesize": PAGE_SIZE,
            "page": page,
            "order": "asc",
            "sort": "creation",
            "filter": filter_id,
        }
        if key:
            params["key"] = key
        response = session.get(f"{API_BASE}/questions", params=params, timeout=60)
        calls += 1
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        if "error_id" in payload:
            raise RuntimeError(json.dumps(payload, ensure_ascii=False)[:500])
        now = datetime.now(timezone.utc).isoformat()
        payload["_metadata"] = {
            "tag_consulted": win.tag_consulted,
            "api_tag": win.api_tag,
            "window_start": win.start.isoformat(),
            "window_end": win.end.isoformat(),
            "fetch_timestamp": now,
            "api_quota_remaining": payload.get("quota_remaining"),
            "page": page,
            "source": SOURCE,
        }
        out_path = raw_page_path(win, page)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        n_items = len(payload.get("items", []))
        items += n_items
        last_quota = payload.get("quota_remaining")
        backoff = payload.get("backoff")
        if backoff:
            time.sleep(float(backoff) + 1.0)
        else:
            time.sleep(polite_sleep)
        if last_quota is not None and int(last_quota) < LOW_QUOTA_STOP:
            raise SystemExit(f"quota_remaining={last_quota}; safe stop")
        if not payload.get("has_more"):
            break
        page += 1
        if page > max_pages:
            page_cap_hit = True
            break
    return {
        "api_calls": calls,
        "items": items,
        "last_quota": last_quota,
        "page_cap_hit": page_cap_hit,
        "pages": page - 1 if page_cap_hit else page,
    }


def run(args: argparse.Namespace) -> None:
    key = args.key or os.environ.get("STACKEXCHANGE_KEY")
    if args.require_key and not key:
        raise SystemExit("STACKEXCHANGE_KEY is not set. Export it and rerun this script.")
    if not key:
        print("[warn] no API key found; use --pilot-tags for small test runs only", file=sys.stderr)

    tags = load_fixed_tags(args.tag_source)
    if args.pilot_tags:
        keep = set(args.pilot_tags)
        tags = [tag for tag in tags if tag in keep]
    force_tags = set(args.force_tags or [])
    targets = query_targets(tags)
    completed = completed_keys(MANIFEST)
    session = requests.Session()
    session.headers.update({"User-Agent": "so-genai-curated-layer/full-2025-extension"})
    filter_id = args.filter_id or create_filter(session, key=key)
    max_pages = MAX_PAGES_WITH_KEY if key else MAX_PAGES_NO_KEY

    queue = []
    for canonical, api_tag in targets:
        level = args.initial_level if canonical in force_tags or api_tag in force_tags else "annual"
        queue.extend(initial_windows(canonical, api_tag, level))
    while queue:
        win = queue.pop(0)
        key_tuple = (win.api_tag, win.start.isoformat(), win.end.isoformat())
        if key_tuple in completed and win.tag_consulted not in force_tags and win.api_tag not in force_tags:
            continue
        print(f"[fetch] {win.api_tag} -> {win.tag_consulted} {win.start.date()} {win.end.date()} ({win.level})", flush=True)
        try:
            stats = fetch_leaf(session, win, key, filter_id, max_pages, args.polite_sleep)
        except SystemExit:
            raise
        except Exception as exc:
            append_csv(FAILED_WINDOWS, {
                "tag_consulted": win.tag_consulted,
                "api_tag": win.api_tag,
                "window_start": win.start.isoformat(),
                "window_end": win.end.isoformat(),
                "level": win.level,
                "error": type(exc).__name__,
                "message": str(exc),
            })
            raise
        if stats["page_cap_hit"]:
            for child in split_window(win):
                queue.insert(0, child)
            append_csv(MANIFEST, {
                "tag_consulted": win.tag_consulted,
                "api_tag": win.api_tag,
                "window_start": win.start.isoformat(),
                "window_end": win.end.isoformat(),
                "level": win.level,
                "status": "split",
                "api_calls": stats["api_calls"],
                "items": stats["items"],
                "quota_remaining": stats["last_quota"],
                "pages": stats["pages"],
                "page_cap_hit": 1,
            })
            continue
        append_csv(MANIFEST, {
            "tag_consulted": win.tag_consulted,
            "api_tag": win.api_tag,
            "window_start": win.start.isoformat(),
            "window_end": win.end.isoformat(),
            "level": win.level,
            "status": "complete",
            "api_calls": stats["api_calls"],
            "items": stats["items"],
            "quota_remaining": stats["last_quota"],
            "pages": stats["pages"],
            "page_cap_hit": 0,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-source", type=Path, default=PROCESSED_DIR / "ai_answerability_index.csv")
    parser.add_argument("--key", default=None)
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument("--polite-sleep", type=float, default=0.25)
    parser.add_argument("--pilot-tags", nargs="*", default=None)
    parser.add_argument(
        "--force-tags",
        nargs="*",
        default=None,
        help="Tags/api-tags to refetch even if complete rows exist in the manifest.",
    )
    parser.add_argument(
        "--initial-level",
        choices=["annual", "semiannual", "monthly", "weekly", "daily"],
        default="annual",
        help="Initial windowing level for force-tags; monthly is useful for repairing suspicious annual API truncation.",
    )
    parser.add_argument("--require-key", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
