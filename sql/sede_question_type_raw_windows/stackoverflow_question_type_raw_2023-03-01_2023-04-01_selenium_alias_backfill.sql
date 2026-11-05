-- Backfill for selenium tag continuity after Stack Overflow tag usage moved
-- toward selenium-webdriver / selenium-chromedriver / webdriver.
-- Save CSV as:
-- stackoverflow_question_type_raw_2023-03-01_2023-04-01_selenium_alias_backfill.csv

SELECT DISTINCT
    'selenium' AS tag,
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
  AND p.CreationDate >= '2023-03-01'
  AND p.CreationDate < '2023-04-01'
  AND t.TagName IN (
      'selenium-webdriver',
      'selenium-chromedriver',
      'webdriver',
      'chromedriver'
  )
ORDER BY
    p.CreationDate,
    p.Id;
