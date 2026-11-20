WITH ranked AS (
    SELECT publisher_id, title,
           ROW_NUMBER() OVER (PARTITION BY publisher_id ORDER BY publication_date, title) AS rn
    FROM raw.book
)
SELECT p.publisher_id,
       p.publisher_name,
       COALESCE(COUNT(b.book_id), 0) AS num_books,
       r.title AS title_of_the_oldest_book,
       ROUND(AVG(b.price), 2) AS avg_price
FROM raw.publisher p
LEFT JOIN raw.book b ON b.publisher_id = p.publisher_id
LEFT JOIN ranked r ON r.publisher_id = p.publisher_id AND r.rn = 1
GROUP BY p.publisher_id, p.publisher_name, r.title
