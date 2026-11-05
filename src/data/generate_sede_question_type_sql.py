"""Generate lightweight Stack Exchange Data Explorer SQL windows.

The query intentionally avoids text classification inside SEDE. It exports
question-tag rows with title and simple structural variables; local Python
scripts classify question types later.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

from src.paths import SQL_DIR


TOP100_TAGS = [
    ".net",
    ".net-core",
    "ajax",
    "algorithm",
    "amazon-web-services",
    "android",
    "android-studio",
    "angular",
    "apache-spark",
    "arrays",
    "asp.net",
    "asp.net-core",
    "asp.net-mvc",
    "azure",
    "bash",
    "c",
    "c#",
    "c++",
    "css",
    "csv",
    "dart",
    "database",
    "dataframe",
    "dictionary",
    "django",
    "docker",
    "excel",
    "express",
    "firebase",
    "flask",
    "flutter",
    "for-loop",
    "function",
    "ggplot2",
    "git",
    "go",
    "google-apps-script",
    "google-cloud-firestore",
    "google-cloud-platform",
    "google-sheets",
    "html",
    "ios",
    "java",
    "javascript",
    "jquery",
    "json",
    "keras",
    "kotlin",
    "kubernetes",
    "laravel",
    "linux",
    "list",
    "loops",
    "machine-learning",
    "macos",
    "matplotlib",
    "mongodb",
    "multithreading",
    "mysql",
    "node.js",
    "numpy",
    "opencv",
    "oracle-database",
    "pandas",
    "php",
    "postgresql",
    "powershell",
    "python",
    "python-3.x",
    "r",
    "react-native",
    "reactjs",
    "regex",
    "rest",
    "ruby",
    "ruby-on-rails",
    "scala",
    "selenium",
    "shell",
    "spring",
    "spring-boot",
    "sql",
    "sql-server",
    "string",
    "swift",
    "swiftui",
    "tensorflow",
    "tkinter",
    "typescript",
    "unity-game-engine",
    "vba",
    "visual-studio",
    "visual-studio-code",
    "vue.js",
    "web-scraping",
    "windows",
    "wordpress",
    "wpf",
    "xcode",
    "xml",
]

TAG_ALIASES = {
    "selenium-webdriver": "selenium",
    "selenium-chromedriver": "selenium",
    "webdriver": "selenium",
    "chromedriver": "selenium",
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def tag_list_sql() -> str:
    tags = TOP100_TAGS + sorted(TAG_ALIASES)
    quoted = [f"'{tag}'" for tag in tags]
    lines = []
    for i in range(0, len(quoted), 7):
        lines.append("    " + ",".join(quoted[i : i + 7]))
    return ",\n".join(lines)


def canonical_tag_sql() -> str:
    cases = "\n".join(
        f"        WHEN t.TagName = '{alias}' THEN '{canonical}'"
        for alias, canonical in sorted(TAG_ALIASES.items())
    )
    return f"""CASE
{cases}
        ELSE t.TagName
    END"""


def build_sql(start: date, end: date) -> str:
    return f"""SELECT DISTINCT
    {canonical_tag_sql()} AS tag,
    DATEADD(WEEK, DATEDIFF(WEEK, 0, p.CreationDate), 0) AS week_start,
    p.Id AS question_id,
    p.OwnerUserId AS owner_user_id,
    p.CreationDate AS creation_date,
    p.Title AS title,
    LEN(COALESCE(p.Body, '')) AS body_length,
    CASE WHEN p.Body LIKE '%<code>%' THEN 1 ELSE 0 END AS has_code,
    p.Score AS score,
    p.AnswerCount AS answer_count,
    CASE WHEN p.AcceptedAnswerId IS NOT NULL THEN 1 ELSE 0 END AS has_accepted_answer,
    CASE WHEN p.ClosedDate IS NOT NULL THEN 1 ELSE 0 END AS is_closed
FROM Posts p
INNER JOIN PostTags pt ON p.Id = pt.PostId
INNER JOIN Tags t ON pt.TagId = t.Id
WHERE p.PostTypeId = 1
  AND p.CreationDate >= '{start.isoformat()}'
  AND p.CreationDate < '{end.isoformat()}'
  AND t.TagName IN (
{tag_list_sql()}
  )
ORDER BY
    creation_date,
    question_id,
    tag;
"""


def window_filename(start: date, end: date, prefix: str) -> str:
    return f"{prefix}_{start.isoformat()}_{end.isoformat()}.sql"


def generate_windows(start: date, end: date, days: int) -> list[tuple[date, date]]:
    windows = []
    current = start
    while current < end:
        next_date = min(current + timedelta(days=days), end)
        windows.append((current, next_date))
        current = next_date
    return windows


def write_sql(start: date, end: date, output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / window_filename(start, end, prefix)
    path.write_text(build_sql(start, end), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Exclusive end date, YYYY-MM-DD.")
    parser.add_argument("--days", type=int, default=4, help="Window size in days.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SQL_DIR / "sede_question_type_raw_windows",
    )
    parser.add_argument("--prefix", default="stackoverflow_question_type_raw")
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    paths = [
        write_sql(window_start, window_end, args.output_dir, args.prefix)
        for window_start, window_end in generate_windows(start, end, args.days)
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
