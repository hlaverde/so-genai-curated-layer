SELECT
    t.TagName AS tag,
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
  AND p.CreationDate >= '2022-05-17'
  AND p.CreationDate < '2022-05-21'
  AND t.TagName IN (
    '.net','.net-core','ajax','algorithm','amazon-web-services','android','android-studio',
    'angular','apache-spark','arrays','asp.net','asp.net-core','asp.net-mvc','azure',
    'bash','c','c#','c++','css','csv','dart',
    'database','dataframe','dictionary','django','docker','excel','express',
    'firebase','flask','flutter','for-loop','function','ggplot2','git',
    'go','google-apps-script','google-cloud-firestore','google-cloud-platform','google-sheets','html','ios',
    'java','javascript','jquery','json','keras','kotlin','kubernetes',
    'laravel','linux','list','loops','machine-learning','macos','matplotlib',
    'mongodb','multithreading','mysql','node.js','numpy','opencv','oracle-database',
    'pandas','php','postgresql','powershell','python','python-3.x','r',
    'react-native','reactjs','regex','rest','ruby','ruby-on-rails','scala',
    'selenium','shell','spring','spring-boot','sql','sql-server','string',
    'swift','swiftui','tensorflow','tkinter','typescript','unity-game-engine','vba',
    'visual-studio','visual-studio-code','vue.js','web-scraping','windows','wordpress','wpf',
    'xcode','xml'
  )
ORDER BY
    p.CreationDate,
    p.Id,
    t.TagName;
