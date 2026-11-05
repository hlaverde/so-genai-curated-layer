-- Stack Overflow question-level validation pool for the AI-answerability validation.
-- Run in Stack Exchange Data Explorer against Stack Overflow.
-- Save as: data/raw/stackoverflow/stackoverflow_question_validation_pool.csv
--
-- Rationale:
--   The project AI-answerability index is built outside SEDE. This query exports
--   a deterministic pre-ChatGPT question pool from top pre-period tags. Python
--   then joins tag-level AI-answerability and draws high/mid/low strata.
--
-- Notes:
--   - p.Id % 100 is a deterministic low-cost sampling rule.
--   - Increase/decrease the modulo if the result is too small/large.
--   - Body is required for the body-length and code features; do not manually edit exports.

WITH TopTags AS (
    SELECT TOP 100
        t.TagName,
        COUNT(*) AS QuestionsPre
    FROM Posts p
    INNER JOIN PostTags pt ON p.Id = pt.PostId
    INNER JOIN Tags t ON pt.TagId = t.Id
    WHERE p.PostTypeId = 1
      AND p.CreationDate >= '2018-01-01'
      AND p.CreationDate < '2022-11-30'
    GROUP BY t.TagName
    ORDER BY COUNT(*) DESC
),
QuestionTags AS (
    SELECT
        p.Id AS question_id,
        COUNT(*) AS num_tags
    FROM Posts p
    INNER JOIN PostTags pt ON p.Id = pt.PostId
    INNER JOIN Tags t ON pt.TagId = t.Id
    INNER JOIN TopTags tt ON t.TagName = tt.TagName
    WHERE p.PostTypeId = 1
      AND p.CreationDate >= '2018-01-01'
      AND p.CreationDate < '2022-11-30'
      AND p.Id % 100 = 0
    GROUP BY p.Id
),
AnswerStats AS (
    SELECT
        ParentId AS question_id,
        COUNT(*) AS answers_observed,
        AVG(CAST(Score AS float)) AS avg_answer_score,
        MIN(CreationDate) AS first_answer_date
    FROM Posts
    WHERE PostTypeId = 2
      AND ParentId IS NOT NULL
    GROUP BY ParentId
)
SELECT
    p.Id AS question_id,
    p.CreationDate AS creation_date,
    p.OwnerUserId AS owner_user_id,
    t.TagName AS tag,
    p.Tags AS all_tags,
    qt.num_tags,
    p.Title AS title,
    p.Body AS body,
    LEN(p.Body) AS body_length,
    CASE WHEN p.Body LIKE '%<code>%' THEN 1 ELSE 0 END AS has_code,
    CASE WHEN LEN(p.Body) < 1200 AND p.Body LIKE '%<code>%' THEN 1 ELSE 0 END AS short_code,
    CASE
        WHEN LOWER(p.Title) LIKE 'how %'
          OR LOWER(p.Title) LIKE 'how do %'
          OR LOWER(p.Title) LIKE 'how to %'
          OR LOWER(p.Title) LIKE '% error %'
          OR LOWER(p.Title) LIKE '% exception %'
        THEN 1 ELSE 0
    END AS how_to_error_title,
    p.Score AS question_score,
    p.AnswerCount AS answer_count,
    CASE WHEN p.AcceptedAnswerId IS NULL THEN 0 ELSE 1 END AS has_accepted_answer,
    p.ClosedDate AS closed_date,
    CASE WHEN p.ClosedDate IS NULL THEN 0 ELSE 1 END AS is_closed,
    a.answers_observed,
    a.avg_answer_score,
    DATEDIFF(minute, p.CreationDate, a.first_answer_date) AS minutes_to_first_answer
FROM Posts p
INNER JOIN QuestionTags qt ON p.Id = qt.question_id
INNER JOIN PostTags pt ON p.Id = pt.PostId
INNER JOIN Tags t ON pt.TagId = t.Id
INNER JOIN TopTags tt ON t.TagName = tt.TagName
LEFT JOIN AnswerStats a ON p.Id = a.question_id
WHERE p.PostTypeId = 1
ORDER BY p.Id, t.TagName;
