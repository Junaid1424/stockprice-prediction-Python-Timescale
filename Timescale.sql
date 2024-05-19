
CREATE TABLE stock_data ( date DATE, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume BIGINT, name TEXT );

SELECT create_hypertable('stock_data', 'date');

ANALYZE stock_data;

SELECT * FROM approximate_row_count('stock_data');

WITH t AS (
    SELECT
        time_bucket('1 day'::interval, "date") AS dt,
        stats_agg(open) AS stats1D
    FROM stock_data
    WHERE Name = 'AAL'
    GROUP BY time_bucket('1 day'::interval, "date")
)
SELECT
    average(stats1D) AS avg_open,
    stddev(stats1D) AS stddev_open,
    skewness(stats1D) AS skewness_open
FROM t;
