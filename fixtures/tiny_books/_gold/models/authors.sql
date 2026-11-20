WITH ab AS (
    SELECT ba.author_id, b.book_id, b.title, b.num_pages, b.publication_date
    FROM raw.book_author ba
    JOIN raw.book b ON b.book_id = ba.book_id
),
first_books AS (
    SELECT author_id, title,
           ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY publication_date, title) AS rn
    FROM ab
),
pages AS (
    SELECT a.author_id, COALESCE(SUM(ab.num_pages), 0) AS total_pages, COUNT(ab.book_id) AS num_books
    FROM raw.author a
    LEFT JOIN ab ON ab.author_id = a.author_id
    GROUP BY a.author_id
)
SELECT a.author_id,
       a.author_name,
       p.num_books,
       f.title AS first_book,
       CASE WHEN p.total_pages > (SELECT AVG(total_pages) FROM pages) THEN 1 ELSE 0 END
           AS wrote_more_pages_than_average
FROM raw.author a
JOIN pages p ON p.author_id = a.author_id
LEFT JOIN first_books f ON f.author_id = a.author_id AND f.rn = 1
