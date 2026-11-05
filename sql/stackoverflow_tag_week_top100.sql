-- Stack Overflow tag-week panel for the top 100 pre-ChatGPT tags.
-- Use this as the preferred SEDE extract if full exports hit the 50,000 row limit.
-- Export as: stackoverflow_tag_week.csv

WITH TopTags AS (
    SELECT TOP 100
        t.TagName,
        COUNT(*) AS PreQuestions
    FROM Posts q
    INNER JOIN PostTags pt ON q.Id = pt.PostId
    INNER JOIN Tags t ON pt.TagId = t.Id
    WHERE q.PostTypeId = 1
      AND q.CreationDate >= '2018-01-01'
      AND q.CreationDate < '2022-11-30'
    GROUP BY t.TagName
    ORDER BY COUNT(*) DESC
),
QuestionTags AS (
    SELECT
        q.Id AS QuestionId,
        q.OwnerUserId,
        DATEADD(day, -DATEDIFF(day, 0, q.CreationDate) % 7, CAST(q.CreationDate AS date)) AS WeekStart,
        q.AcceptedAnswerId,
        q.Score,
        q.ClosedDate,
        q.AnswerCount,
        t.TagName
    FROM Posts q
    INNER JOIN PostTags pt ON q.Id = pt.PostId
    INNER JOIN Tags t ON pt.TagId = t.Id
    INNER JOIN TopTags tt ON t.TagName = tt.TagName
    WHERE q.PostTypeId = 1
      AND q.CreationDate >= '2018-01-01'
)
SELECT
    qt.TagName AS tag,
    qt.WeekStart AS week_start,
    COUNT(*) AS questions,
    SUM(COALESCE(qt.AnswerCount, 0)) AS answers,
    SUM(CASE WHEN qt.AcceptedAnswerId IS NOT NULL THEN 1 ELSE 0 END) AS accepted_answers,
    AVG(CASE WHEN qt.AnswerCount > 0 THEN 1.0 ELSE 0.0 END) AS answer_rate,
    AVG(CAST(qt.Score AS float)) AS avg_score,
    SUM(CASE WHEN qt.ClosedDate IS NOT NULL THEN 1 ELSE 0 END) AS closed_questions,
    COUNT(DISTINCT qt.OwnerUserId) AS unique_users
FROM QuestionTags qt
GROUP BY qt.TagName, qt.WeekStart
ORDER BY qt.TagName, qt.WeekStart;
