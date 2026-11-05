# Stack Exchange Data Explorer (SEDE) download protocol

The extracts up to May 2023 come from the Stack Exchange Data Explorer,
which runs queries only in an interactive browser session. The query text
is versioned in `sql/`; each query is run by hand and its CSV is saved
unedited under `data/raw/stackoverflow/` with the name given below.

## Queries and file names

| Query | Save as | Used for |
|---|---|---|
| `sql/sede_question_type_raw_windows/stackoverflow_question_type_raw_{start}_{end}.sql` (one file per four-day window, 2020-01 to 2023-06) | `stackoverflow_question_type_raw_{start}_{end}.csv` | Question–tag rows of the main panel |
| `sql/stackoverflow_tag_week_top100.sql` | `stackoverflow_tag_week.csv` | Tag-week aggregates for the answerability index |
| `sql/sede_windows/stackoverflow_post_complexity_tag_week_window.sql`, run with `##StartDate##`/`##EndDate##` set to 2018-01-01/2020-12-31, 2021-01-01/2023-12-31 and 2024-01-01/2026-12-31 | `stackoverflow_post_complexity_tag_week_2018_2020.csv`, `..._2021_2023.csv`, `..._2024_2026.csv` | Tag-week complexity aggregates for the answerability index |
| `sql/stackoverflow_question_validation_pool.sql` | `stackoverflow_question_validation_pool.csv` | Question-level sample for the classifier and revealed-answerability validations |

The window queries are written by
`python -m src.data.generate_sede_question_type_sql --start 2020-01-01 --end 2023-07-01`.
From June 2023 the same rows come from the Stack Exchange API
(`src/data/fetch_stackoverflow_via_api.py`); question–tag pairs present in
both sources are counted once by the pipeline.

## Browser steps

1. Open <https://data.stackexchange.com/stackoverflow/query/new> and log in.
2. Paste the query, set its parameters if it has any, and run it.
3. Click **Download CSV** and save the file under
   `data/raw/stackoverflow/` with the name above.
4. Do not open and re-save the file in a spreadsheet program.

## Notes

- SEDE refreshes its copy of the database periodically; scores and
  answer counts of recent posts change between refreshes.
- SEDE assigns Sunday questions to the following Monday's week; the
  pipeline recomputes Monday-to-Sunday weeks from the UTC creation date,
  so the export's own week column is not used.
