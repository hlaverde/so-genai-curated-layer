# Analysis data

Every input of the estimation, as gzip-compressed CSV. `python run_pipeline.py
--from-analysis-data` starts from these files. They hold no title, body or other
question text and no user identifier. They are derived from Stack Overflow content
and distributed under CC BY-SA 4.0 (see `LICENSE`).

| File | Level | Rows | Content |
|---|---|---|---|
| `panel_main.csv.gz` | Tag × week × question type | 182,700 | Main panel, 2020-01-06 to 2025-01-05 |
| `panel_extended.csv.gz` | Tag × week × question type | 218,400 | Main panel plus 2025, to 2025-12-28 |
| `question_tag.csv.gz` | Question × tag | 8,511,130 | Every question–tag pair of the complete weeks |
| `minilm_panel_title_labels.csv.gz` | Question | 4,987,056 | MiniLM label of every question of the main panel |
| `validation_sample.csv.gz` | Question | 9,996 | Labels of the three classifiers in the validation sample |
| `revealed_pool.csv.gz` | Question × tag | 50,000 | Pool of the revealed-answerability validation |
| `selenium_backfill.csv.gz` | Question | 793 | Questions of the `selenium` alias query |
| `tag_index.csv.gz` | Tag | 100 | Components of the tag index, the index and its variants |
| `tag_embedding_score.csv.gz` | Tag | 100 | Embedding answerability by tag |
| `question_type_substitutability.csv.gz` | Question type | 7 | Continuous substitutability score |
| `weeks.csv.gz` | Week | 312 | Complete weeks |
| `dropped_weeks.csv.gz` | Week | 2 | Partial weeks left out of every panel |
| `collection_counts.csv.gz` | Statistic | 11 | Counts from the raw files |

The monitoring procedure and the 2025 extension use `panel_main.csv.gz`,
`panel_extended.csv.gz` and `tag_index.csv.gz`.

## Cell panels

One row per tag, week and question type on the full grid, with zero counts in cells
that have no question. Counts are of question–tag pairs in the cell.

| Variable | Definition |
|---|---|
| `tag`, `week_start`, `question_type` | Cell: tag, Monday of the Monday-to-Sunday week (UTC), question type |
| `questions` | Questions |
| `answered` | Questions with at least one answer |
| `accepted` | Questions with an accepted answer |
| `acc_nonclosed` | Accepted and not closed |
| `acc_nonneg` | Accepted with a non-negative score |
| `funnel` | Funnel-qualified posts: answered, accepted, not closed, non-negative score |
| `s0` to `s4`, `s5p` | Funnel-qualified posts by score: 0 to 4, and 5 or more |
| `ge0` to `ge5` | Funnel-qualified posts with a score at or above the number |
| `value_w` | Sum of log(1 + score) over accepted, non-closed questions |
| `q_frac` | Questions weighted by 1/k, with k the number of panel tags on the question |
| `answers` | Sum of the answer counts |
| `closed` | Closed questions |
| `code_q` | Questions with a `<code>` element |
| `unique_users` | Distinct question authors |
| `body_length_mean` | Mean body length in characters; empty in a cell with no question |
| `avg_score` | Mean score; empty in a cell with no question |
| `sub` | 1 if the question type is substitutable |
| `vn1`, `vn2`, `vn5` | Main panel: accepted, non-closed questions with a voting-normalized score at or above the number, with the tag's pre/post ratio clipped to [0.5, 3] |
| suffix `_noclip`, `_c35`, `_c72` | The same without clipping, and clipped to [0.3, 5] and [0.7, 2] |

## Question–tag rows

| Variable | Definition |
|---|---|
| `question_id` | Stack Overflow question ID |
| `tag` | Tag, with the `selenium` aliases counted under `selenium` |
| `day` | UTC creation date |
| `question_type` | Question type of the rule in `src/data/prepare_stackoverflow_question_type_raw.py` |
| `control_mask` | Bit j set when the title contains `CONTROL_WORDS[j]` of that file; zero for a substitutable title |
| `score` | Score at collection |
| `curated` | 1 if the question has an accepted answer and is not closed |

## Other files

| File | Variables |
|---|---|
| `minilm_panel_title_labels.csv.gz` | `question_id`; `minilm_type`, the nearest category centroid of `all-MiniLM-L6-v2` |
| `validation_sample.csv.gz` | `question_id`, `body_length`, `has_code`; `heuristic_type`, `minilm_type`, `mpnet_type`, the label of each classifier |
| `revealed_pool.csv.gz` | `question_id`, `tag`, `answer_count`, `has_accepted_answer`, `is_closed`, `minutes_to_first_answer` |
| `selenium_backfill.csv.gz` | `question_id`, `creation_date` |
| `tag_index.csv.gz` | `tag`; pre-ChatGPT `questions_pre`, `accepted_answers_pre`, `answer_rate_pre`, `historical_frequency_pre`, `tag_maturity_weeks_pre`, `accepted_answer_rate_pre`, `short_code_share_pre`, `how_to_share_pre`; `ai_answerability_structural` (the tag index) and its variants `ai_answerability_zscore`, `ai_answerability_pca`, `ai_answerability_quantile` |
| `tag_embedding_score.csv.gz` | `tag`; `embed`, the mean similarity of up to 1,000 random pre-ChatGPT titles of the tag to fixed exemplar questions; `n_titles`, the number of titles |
| `question_type_substitutability.csv.gz` | `question_type`, `score` |
| `weeks.csv.gz` | `week_start`; `main_window`, 1 if the week is in the main panel |
| `dropped_weeks.csv.gz` | `week_start`; `days`, the days of the week observed |
| `collection_counts.csv.gz` | `key`, `value`: the comparison of the API and SEDE for 2023-05-02 (`Val*`), the questions collected for 2025 (`NCollectedTwentyFive`), and the window of the tag index (`IndexWinStart`, `IndexWinEnd`) |
