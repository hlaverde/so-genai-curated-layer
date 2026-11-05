# Stack Exchange Data Explorer queries

These queries run by hand in the Stack Exchange Data Explorer against the
Stack Overflow database; `docs/stackoverflow_sede_download.md` gives the
file name under which each CSV is saved and how it enters the analysis.

| Query | Output file |
|---|---|
| `sede_question_type_raw_windows/*.sql` | `stackoverflow_question_type_raw_{start}_{end}.csv` |
| `stackoverflow_tag_week_top100.sql` | `stackoverflow_tag_week.csv` |
| `sede_windows/stackoverflow_post_complexity_tag_week_window.sql` | `stackoverflow_post_complexity_tag_week_{period}.csv` |
| `stackoverflow_question_validation_pool.sql` | `stackoverflow_question_validation_pool.csv` |
