-- Parameterized Stack Overflow tag-week complexity extract for top 100 pre-ChatGPT tags.
-- Use when the all-period complexity query times out.
-- Export as: stackoverflow_post_complexity_tag_week_YYYY_YYYY.csv

##StartDate:string?2021-01-01##
##EndDate:string?2023-12-31##

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
        DATEADD(day, -DATEDIFF(day, 0, q.CreationDate) % 7, CAST(q.CreationDate AS date)) AS WeekStart,
        t.TagName,
        LEN(q.Body) AS body_length,
        CASE WHEN q.Body LIKE '%<code>%' THEN 1 ELSE 0 END AS has_code,
        LEN(q.Tags) - LEN(REPLACE(q.Tags, '<', '')) AS num_tags,
        q.AnswerCount AS answer_count,
        CASE WHEN q.AcceptedAnswerId IS NOT NULL THEN 1 ELSE 0 END AS has_accepted_answer,
        q.Score AS score,
        CASE
            WHEN q.Title LIKE '%how to%'
              OR q.Title LIKE '%How to%'
              OR q.Title LIKE '%how do I%'
              OR q.Title LIKE '%How do I%'
              OR q.Title LIKE '%how can I%'
              OR q.Title LIKE '%How can I%'
              OR q.Title LIKE '%error%'
              OR q.Title LIKE '%Error%'
              OR q.Title LIKE '%fix%'
              OR q.Title LIKE '%Fix%'
            THEN 1 ELSE 0
        END AS how_to_question,
        CASE
            WHEN q.Body LIKE '%<code>%'
             AND LEN(q.Body) <= 750
            THEN 1 ELSE 0
        END AS short_code_question
    FROM Posts q
    INNER JOIN PostTags pt ON q.Id = pt.PostId
    INNER JOIN Tags t ON pt.TagId = t.Id
    INNER JOIN TopTags tt ON t.TagName = tt.TagName
    WHERE q.PostTypeId = 1
      AND q.CreationDate >= CAST(##StartDate## AS datetime)
      AND q.CreationDate < DATEADD(day, 1, CAST(##EndDate## AS datetime))
)
SELECT
    TagName AS tag,
    WeekStart AS week_start,
    COUNT(*) AS questions,
    AVG(CAST(body_length AS float)) AS avg_body_length,
    AVG(CAST(has_code AS float)) AS code_share,
    AVG(CAST(num_tags AS float)) AS avg_num_tags,
    AVG(CAST(answer_count AS float)) AS avg_answer_count,
    AVG(CAST(has_accepted_answer AS float)) AS accepted_answer_share,
    AVG(CAST(score AS float)) AS avg_score,
    AVG(CAST(how_to_question AS float)) AS how_to_share,
    AVG(CAST(short_code_question AS float)) AS short_code_share
FROM QuestionTags
GROUP BY TagName, WeekStart
ORDER BY TagName, WeekStart;
