-- clickhouse-init/init.sql
-- Sample schema for analytics database

CREATE DATABASE IF NOT EXISTS analytics;

USE analytics;

-- Sales data table (example)
CREATE TABLE IF NOT EXISTS sales (
    id UInt64,
    date Date,
    product_name String,
    category String,
    revenue Float64,
    quantity UInt32,
    region String,
    quarter UInt8,
    year UInt16
) ENGINE = MergeTree()
ORDER BY (date, id);

-- Customer data table
CREATE TABLE IF NOT EXISTS customers (
    customer_id UInt64,
    customer_name String,
    email String,
    registration_date Date,
    total_purchases Float64,
    region String,
    customer_segment String
) ENGINE = MergeTree()
ORDER BY customer_id;

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    timestamp DateTime,
    level String,
    service String,
    message String,
    user_id Nullable(UInt64)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY timestamp;

-- Insert sample data for testing
INSERT INTO sales VALUES
    (1, '2024-01-15', 'Product A', 'Electronics', 150000, 100, 'North', 1, 2024),
    (2, '2024-02-20', 'Product B', 'Electronics', 200000, 150, 'South', 1, 2024),
    (3, '2024-03-10', 'Product C', 'Furniture', 180000, 80, 'East', 1, 2024),
    (4, '2024-04-05', 'Product A', 'Electronics', 220000, 120, 'West', 2, 2024),
    (5, '2024-05-18', 'Product D', 'Appliances', 300000, 200, 'North', 2, 2024),
    (6, '2024-06-22', 'Product B', 'Electronics', 250000, 180, 'South', 2, 2024),
    (7, '2024-07-30', 'Product E', 'Furniture', 190000, 90, 'East', 3, 2024),
    (8, '2024-08-14', 'Product A', 'Electronics', 280000, 140, 'North', 3, 2024),
    (9, '2024-09-05', 'Product C', 'Furniture', 210000, 100, 'West', 3, 2024),
    (10, '2024-10-11', 'Product D', 'Appliances', 350000, 250, 'South', 4, 2024),
    (11, '2024-11-20', 'Product B', 'Electronics', 290000, 200, 'North', 4, 2024),
    (12, '2024-12-15', 'Product E', 'Furniture', 240000, 110, 'East', 4, 2024);

INSERT INTO customers VALUES
    (1, 'Customer A', 'a@example.com', '2023-01-15', 450000, 'North', 'Premium'),
    (2, 'Customer B', 'b@example.com', '2023-03-20', 280000, 'South', 'Standard'),
    (3, 'Customer C', 'c@example.com', '2023-06-10', 620000, 'East', 'Premium'),
    (4, 'Customer D', 'd@example.com', '2023-09-05', 190000, 'West', 'Basic'),
    (5, 'Customer E', 'e@example.com', '2024-01-12', 520000, 'North', 'Premium');

-- Create views for common queries
CREATE VIEW IF NOT EXISTS quarterly_revenue AS
SELECT 
    year,
    quarter,
    region,
    sum(revenue) as total_revenue,
    sum(quantity) as total_quantity
FROM sales
GROUP BY year, quarter, region
ORDER BY year, quarter, region;

CREATE VIEW IF NOT EXISTS category_performance AS
SELECT 
    category,
    count(*) as num_sales,
    sum(revenue) as total_revenue,
    avg(revenue) as avg_revenue,
    sum(quantity) as total_quantity
FROM sales
GROUP BY category
ORDER BY total_revenue DESC;

-- Grant permissions (if needed)
-- GRANT SELECT ON analytics.* TO default;