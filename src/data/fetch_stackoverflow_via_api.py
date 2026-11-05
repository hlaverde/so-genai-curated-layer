"""
Download Stack Overflow questions through the Stack Exchange API v2.3.

Writes one CSV per window from the public /questions endpoint.

Output schema (the same as the SEDE extracts):
    tag, week_start, question_id, owner_user_id, creation_date, title,
    body_length, has_code, score, answer_count, has_accepted_answer, is_closed

Usage:
    # Without a key (300 requests per day per IP, for small test runs only):
    python -m src.data.fetch_stackoverflow_via_api \
        --start 2023-06-01 --end 2023-06-05

    # With a free key registered at https://stackapps.com/apps/oauth/register
    # (10,000 requests per day):
    export STACKEXCHANGE_KEY=your_key_here
    python -m src.data.fetch_stackoverflow_via_api \
        --start 2023-06-01 --end 2023-07-01 --days 4

The script uses only public endpoints of api.stackexchange.com and honors the
backoff the API requests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests


API_BASE = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"


# Canonical top 100 tags (without the selenium aliases)
TOP_100_TAGS: set[str] = {
    ".net", ".net-core", "ajax", "algorithm", "amazon-web-services",
    "android", "android-studio", "angular", "apache-spark", "arrays",
    "asp.net", "asp.net-core", "asp.net-mvc", "azure", "bash",
    "c", "c#", "c++", "css", "csv",
    "dart", "database", "dataframe", "dictionary", "django",
    "docker", "excel", "express", "firebase", "flask",
    "flutter", "for-loop", "function", "ggplot2", "git",
    "go", "google-apps-script", "google-cloud-firestore",
    "google-cloud-platform", "google-sheets", "html", "ios", "java",
    "javascript", "jquery", "json", "keras", "kotlin", "kubernetes",
    "laravel", "linux", "list", "loops", "machine-learning", "macos",
    "matplotlib", "mongodb", "multithreading", "mysql", "node.js",
    "numpy", "opencv", "oracle-database", "pandas", "php", "postgresql",
    "powershell", "python", "python-3.x", "r", "react-native", "reactjs",
    "regex", "rest", "ruby", "ruby-on-rails", "scala", "selenium",
    "shell", "spring", "spring-boot", "sql", "sql-server", "string",
    "swift", "swiftui", "tensorflow", "tkinter", "typescript",
    "unity-game-engine", "vba", "visual-studio", "visual-studio-code",
    "vue.js", "web-scraping", "windows", "wordpress", "wpf",
    "xcode", "xml",
}

SELENIUM_ALIASES: dict[str, str] = {
    "selenium-webdriver": "selenium",
    "selenium-chromedriver": "selenium",
    "webdriver": "selenium",
    "chromedriver": "selenium",
}

ACCEPTED_TAGS: set[str] = TOP_100_TAGS | set(SELENIUM_ALIASES.keys())

OUTPUT_COLUMNS = [
    "tag", "week_start", "question_id", "owner_user_id", "creation_date",
    "title", "body_length", "has_code", "score", "answer_count",
    "has_accepted_answer", "is_closed",
]


def to_canonical(tag: str) -> str:
    return SELENIUM_ALIASES.get(tag, tag)


def utc_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def week_start_of(dt: datetime) -> str:
    """Return Monday 00:00:00 of the Monday-to-Sunday week of dt (UTC).

    This does not replicate DATEADD(WEEK, DATEDIFF(WEEK, 0, dt), 0) of SQL
    Server: DATEDIFF counts Sunday crossings, so in the SEDE extracts a Sunday
    falls in the week of the following Monday. Here a Sunday belongs to the week
    of the preceding Monday. The pipeline recomputes the week from the creation
    date for every source.
    """
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d %H:%M:%S").split(" ")[0] + " 00:00:00"


# Pre-created filter that adds question.body + .total + .has_more +
# .quota_remaining + .backoff to the "default" base. Stack Exchange always
# returns the same ID for the same parameters, so this value is stable and
# reproducible. It is the fallback when creating a filter is throttled.
DEFAULT_FILTER_ID = "!20aKG._8Oscv*6djt*X3)"


def create_filter(session: requests.Session, key: str | None = None) -> str:
    """Filter that extends 'default' with body, total, has_more, backoff.

    Pass `key` so the call is charged to the application quota (10,000 per day)
    instead of the IP quota (300 per day). If the API still answers with a
    throttle violation, the known pre-created filter is used.
    """
    params = {
        "include": "question.body;.total;.has_more;.quota_remaining;.backoff",
        "base": "default",
        "unsafe": "false",
    }
    if key:
        params["key"] = key
    try:
        r = session.get(f"{API_BASE}/filters/create", params=params, timeout=30)
        if r.status_code == 400:
            data = r.json()
            if data.get("error_name") == "throttle_violation":
                print(
                    f"  [warn] /filters/create throttled, using the cached filter "
                    f"({DEFAULT_FILTER_ID})",
                    file=sys.stderr,
                )
                return DEFAULT_FILTER_ID
        r.raise_for_status()
        data = r.json()
        if "items" not in data or not data["items"]:
            raise RuntimeError(f"Filter creation failed: {data}")
        return data["items"][0]["filter"]
    except requests.HTTPError:
        raise


def question_to_rows(q: dict) -> list[dict]:
    tags_raw = q.get("tags") or []
    canonical = set()
    for t in tags_raw:
        if t in ACCEPTED_TAGS:
            canonical.add(to_canonical(t))
    if not canonical:
        return []

    creation_ts = q.get("creation_date")
    if creation_ts is None:
        return []
    creation_dt = datetime.fromtimestamp(creation_ts, tz=timezone.utc)
    creation_str = creation_dt.strftime("%Y-%m-%d %H:%M:%S")
    week_str = week_start_of(creation_dt)

    body = q.get("body") or ""
    title = q.get("title") or ""
    owner = q.get("owner") or {}
    owner_uid = owner.get("user_id", "")

    common = {
        "week_start": week_str,
        "question_id": q.get("question_id"),
        "owner_user_id": owner_uid,
        "creation_date": creation_str,
        "title": title,
        "body_length": len(body),
        "has_code": 1 if "<code>" in body else 0,
        "score": int(q.get("score", 0)),
        "answer_count": int(q.get("answer_count", 0)),
        "has_accepted_answer": 1 if q.get("accepted_answer_id") else 0,
        "is_closed": 1 if q.get("closed_date") else 0,
    }
    return [{"tag": tag, **common} for tag in sorted(canonical)]


class PageLimitExceeded(Exception):
    """The API caps the number of pages (25 without a key, about 200 with one)."""


def _fetch_paginated(
    session: requests.Session,
    start_dt: datetime,
    end_dt: datetime,
    filter_id: str,
    key: str | None,
    max_pages: int,
    polite_sleep: float,
    verbose: bool,
) -> tuple[list[dict], dict]:
    """Linear pagination in a sub-window. Raises PageLimitExceeded if has_more is
    still True on reaching max_pages (the signal to split the window)."""
    rows: list[dict] = []
    page = 1
    questions_seen = 0
    requests_made = 0
    last_quota = None
    fromdate = utc_ts(start_dt)
    todate = utc_ts(end_dt)

    while True:
        params = {
            "site": SITE,
            "fromdate": fromdate,
            "todate": todate,
            "page": page,
            "pagesize": 100,
            "order": "asc",
            "sort": "creation",
            "filter": filter_id,
        }
        if key:
            params["key"] = key
        r = session.get(f"{API_BASE}/questions", params=params, timeout=60)
        requests_made += 1
        # A 400 can mean (a) page limit exceeded or (b) throttle violation.
        # The body tells them apart.
        if r.status_code == 400:
            try:
                err = r.json()
            except Exception:
                err = {}
            ename = err.get("error_name", "")
            if ename == "throttle_violation":
                # Wait reported by the API, if present in error_message
                msg = err.get("error_message", "")
                wait_s = 60
                # Messages like "more requests available in X seconds"
                import re
                m = re.search(r"in (\d+) seconds", msg)
                if m:
                    wait_s = min(int(m.group(1)) + 5, 600)
                if verbose:
                    print(
                        f"  [throttle] waiting {wait_s}s before retrying "
                        f"page={page}",
                        file=sys.stderr,
                    )
                time.sleep(wait_s)
                continue
            if page > 1:
                # Taken as a page-limit hit; signal to split
                raise PageLimitExceeded(
                    f"HTTP 400 at page={page} window {start_dt}-{end_dt}"
                )
            # Any other 400
            raise RuntimeError(f"HTTP 400 at page=1: {err}")
        if r.status_code == 429:
            if verbose:
                print("  HTTP 429, sleeping 30s...", file=sys.stderr)
            time.sleep(30)
            continue
        r.raise_for_status()
        data = r.json()
        if "error_id" in data:
            raise RuntimeError(f"API error: {data}")

        for q in data.get("items", []):
            questions_seen += 1
            rows.extend(question_to_rows(q))

        last_quota = data.get("quota_remaining")
        backoff = data.get("backoff")
        if verbose:
            print(
                f"    [{start_dt:%m-%d %H:%M}-{end_dt:%m-%d %H:%M}] "
                f"page={page} +{len(data.get('items', []))} "
                f"rows={len(rows)} quota={last_quota} backoff={backoff or 0}",
                file=sys.stderr,
            )
        if backoff:
            time.sleep(float(backoff) + 0.2)
        else:
            time.sleep(polite_sleep)

        if not data.get("has_more"):
            break
        page += 1
        if page > max_pages:
            raise PageLimitExceeded(
                f"max_pages={max_pages} reached in {start_dt}-{end_dt} with has_more=True"
            )

    return rows, {
        "requests_made": requests_made,
        "questions_seen": questions_seen,
        "rows_emitted": len(rows),
        "quota_remaining": last_quota,
    }


def fetch_window(
    session: requests.Session,
    start_dt: datetime,
    end_dt: datetime,
    filter_id: str,
    key: str | None,
    polite_sleep: float = 0.25,
    verbose: bool = True,
    sub_window_hours: int = 4,
    min_sub_window_minutes: int = 5,
) -> tuple[list[dict], dict]:
    """Download [start_dt, end_dt) with adaptive sub-windows.

    Without a key the API allows 25 pages (2,500 items); with a key about 200
    (20,000 items). To stay under the limit, the window is divided into
    sub-windows of `sub_window_hours`, and any that exceeds it is halved
    recursively down to `min_sub_window_minutes`.
    """
    max_pages_no_key = 24
    max_pages_with_key = 199
    max_pages = max_pages_with_key if key else max_pages_no_key

    rows: list[dict] = []
    stats = {"requests_made": 0, "questions_seen": 0, "rows_emitted": 0, "quota_remaining": None}

    def _recurse(s: datetime, e: datetime) -> None:
        try:
            r, st = _fetch_paginated(
                session, s, e, filter_id, key, max_pages, polite_sleep, verbose
            )
            rows.extend(r)
            stats["requests_made"] += st["requests_made"]
            stats["questions_seen"] += st["questions_seen"]
            stats["rows_emitted"] += st["rows_emitted"]
            stats["quota_remaining"] = st["quota_remaining"]
        except PageLimitExceeded as exc:
            span = e - s
            if span <= timedelta(minutes=min_sub_window_minutes):
                raise RuntimeError(
                    f"Minimum window ({min_sub_window_minutes}min) exceeded without exhausting pagination: {exc}"
                )
            mid = s + span / 2
            if verbose:
                print(
                    f"  [split] {s:%m-%d %H:%M}-{e:%m-%d %H:%M} -> "
                    f"{s:%H:%M}/{mid:%H:%M}/{e:%H:%M}",
                    file=sys.stderr,
                )
            _recurse(s, mid)
            _recurse(mid, e)

    # Initial sub-windows of `sub_window_hours`
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(hours=sub_window_hours), end_dt)
        _recurse(cur, nxt)
        cur = nxt

    return rows, stats


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Same order as ORDER BY p.CreationDate, p.Id, t.TagName in the SEDE query.
    rows_sorted = sorted(rows, key=lambda r: (r["creation_date"], r["question_id"], r["tag"]))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for row in rows_sorted:
            w.writerow(row)


def daterange_windows(start: datetime, end: datetime, days: int) -> Iterable[tuple[datetime, datetime]]:
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD exclusive")
    p.add_argument("--days", type=int, default=4, help="Window length in days per file")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/stackoverflow"),
        help="Destination folder (raw)",
    )
    p.add_argument("--polite-sleep", type=float, default=0.25)
    p.add_argument(
        "--key",
        default=os.environ.get("STACKEXCHANGE_KEY"),
        help="Stack Exchange key (also through the environment variable STACKEXCHANGE_KEY)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing CSV",
    )
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "so-genai-curated-layer/research (academic, contact: replication)",
            "Accept": "application/json",
        }
    )

    if not args.key:
        print(
            "[WARN] STACKEXCHANGE_KEY is not set. Quota=300 requests/IP/day.\n"
            "       For large downloads register a free application at\n"
            "       https://stackapps.com/apps/oauth/register and export\n"
            "       STACKEXCHANGE_KEY before running.",
            file=sys.stderr,
        )

    print("[1/2] Creating the API filter...", file=sys.stderr)
    filter_id = create_filter(session, key=args.key)
    print(f"       Filter ID: {filter_id}", file=sys.stderr)

    print("[2/2] Downloading windows...", file=sys.stderr)
    total_stats = {"requests": 0, "questions": 0, "rows": 0, "files": 0}
    for win_start, win_end in daterange_windows(start, end, args.days):
        name = (
            f"stackoverflow_question_type_raw_"
            f"{win_start:%Y-%m-%d}_{win_end:%Y-%m-%d}.csv"
        )
        out_path = args.out_dir / name
        if out_path.exists() and not args.force:
            print(f"  [skip] {out_path.name} already exists (use --force to overwrite)",
                  file=sys.stderr)
            continue
        print(f"  Window {win_start:%Y-%m-%d} -> {win_end:%Y-%m-%d}", file=sys.stderr)
        rows, stats = fetch_window(
            session, win_start, win_end, filter_id, args.key, args.polite_sleep
        )
        write_csv(rows, out_path)
        print(
            f"  -> {out_path.name}: rows={stats['rows_emitted']} "
            f"questions={stats['questions_seen']} "
            f"requests={stats['requests_made']} "
            f"quota_remaining={stats['quota_remaining']}",
            file=sys.stderr,
        )
        total_stats["requests"] += stats["requests_made"]
        total_stats["questions"] += stats["questions_seen"]
        total_stats["rows"] += stats["rows_emitted"]
        total_stats["files"] += 1

    print(f"\nDone. {json.dumps(total_stats)}", file=sys.stderr)


if __name__ == "__main__":
    main()
