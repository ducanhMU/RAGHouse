CREATE DATABASE IF NOT EXISTS analytics;
USE analytics;

CREATE TABLE IF NOT EXISTS dim_company (
    company_key         BIGINT NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    company_id          VARCHAR(36),
    company_name_vn     VARCHAR(255),
    company_name_en     VARCHAR(255),
    industry_gics       VARCHAR(100),
    sector              VARCHAR(100),
    exchange            VARCHAR(20),
    listing_date        DATE,
    shares_outstanding  DECIMAL(20, 0) DEFAULT "0",
    free_float          DECIMAL(5, 4) DEFAULT "0",
    foreign_room        DECIMAL(5, 4) DEFAULT "0",
    is_active           TINYINT DEFAULT "1",
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_company_symbol (symbol) USING BITMAP
) ENGINE = OLAP
PRIMARY KEY (company_key)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "enable_persistent_index" = "true",
    "compression" = "LZ4"
);

CREATE TABLE IF NOT EXISTS dim_period (
    period_key          BIGINT NOT NULL,
    year                SMALLINT NOT NULL,
    quarter             TINYINT NOT NULL,
    period_type         VARCHAR(10),
    start_date          DATE,
    end_date            DATE,
    is_latest_quarter   TINYINT DEFAULT "0",
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_period_year (year) USING BITMAP,
    INDEX idx_period_quarter (quarter) USING BITMAP
) ENGINE = OLAP
PRIMARY KEY (period_key)
DISTRIBUTED BY HASH(period_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

CREATE TABLE IF NOT EXISTS dim_period_mapping (
    current_period_key  BIGINT NOT NULL,
    prev_period_key     BIGINT,
    year_diff           TINYINT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (current_period_key)
DISTRIBUTED BY HASH(current_period_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

CREATE TABLE IF NOT EXISTS fact_income_statement (
    company_key                     BIGINT NOT NULL,
    period_key                      BIGINT NOT NULL,
    report_date                     DATE NOT NULL,
    revenue                         DECIMAL(24, 2) DEFAULT "0",
    cogs                            DECIMAL(24, 2) DEFAULT "0",
    gross_profit                    DECIMAL(24, 2) DEFAULT "0",
    sgna                            DECIMAL(24, 2) DEFAULT "0",
    operating_profit                DECIMAL(24, 2) DEFAULT "0",
    financial_income                DECIMAL(24, 2) DEFAULT "0",
    financial_expense               DECIMAL(24, 2) DEFAULT "0",
    interest_expense                DECIMAL(24, 2) DEFAULT "0",
    lease_interest_expense          DECIMAL(24, 2) DEFAULT "0",
    profit_before_tax               DECIMAL(24, 2) DEFAULT "0",
    tax                             DECIMAL(24, 2) DEFAULT "0",
    effective_tax_rate              DECIMAL(7, 4) DEFAULT "0",
    net_income                      DECIMAL(24, 2) DEFAULT "0",
    eps                             DECIMAL(18, 4) DEFAULT "0",
    eps_diluted                     DECIMAL(18, 4) DEFAULT "0",
    ebitda                          DECIMAL(24, 2) DEFAULT "0",
    depreciation_amortization_total DECIMAL(24, 2) DEFAULT "0",
    ebit                            DECIMAL(24, 2) DEFAULT "0",
    raw_json                        JSON,
    created_at                      DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_is_company (company_key) USING BITMAP,
    INDEX idx_is_period (period_key) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (company_key, period_key, report_date)
PARTITION BY RANGE(report_date) () 
DISTRIBUTED BY HASH(company_key) BUCKETS 8
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

CREATE TABLE IF NOT EXISTS fact_balance_sheet (
    company_key                 BIGINT NOT NULL,
    period_key                  BIGINT NOT NULL,
    report_date                 DATE NOT NULL,
    cash                        DECIMAL(24, 2) DEFAULT "0",
    short_term_invest           DECIMAL(24, 2) DEFAULT "0",
    receivables                 DECIMAL(24, 2) DEFAULT "0",
    inventory                   DECIMAL(24, 2) DEFAULT "0",
    total_current_assets        DECIMAL(24, 2) DEFAULT "0",
    fixed_assets                DECIMAL(24, 2) DEFAULT "0",
    lease_assets                DECIMAL(24, 2) DEFAULT "0",
    intangible_assets           DECIMAL(24, 2) DEFAULT "0",
    total_assets                DECIMAL(24, 2) DEFAULT "0",
    payables                    DECIMAL(24, 2) DEFAULT "0",
    short_term_debt             DECIMAL(24, 2) DEFAULT "0",
    lease_liabilities_current   DECIMAL(24, 2) DEFAULT "0",
    long_term_debt              DECIMAL(24, 2) DEFAULT "0",
    lease_liabilities_long      DECIMAL(24, 2) DEFAULT "0",
    total_current_liab          DECIMAL(24, 2) DEFAULT "0",
    total_liabilities           DECIMAL(24, 2) DEFAULT "0",
    share_capital               DECIMAL(24, 2) DEFAULT "0",
    retained_earnings           DECIMAL(24, 2) DEFAULT "0",
    total_equity                DECIMAL(24, 2) DEFAULT "0",
    bvps                        DECIMAL(18, 4) DEFAULT "0",
    raw_json                    JSON,
    created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_bs_company (company_key) USING BITMAP,
    INDEX idx_bs_period (period_key) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (company_key, period_key, report_date)
PARTITION BY RANGE(report_date) () 
DISTRIBUTED BY HASH(company_key) BUCKETS 8
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

CREATE TABLE IF NOT EXISTS fact_cash_flow (
    company_key                 BIGINT NOT NULL,
    period_key                  BIGINT NOT NULL,
    report_date                 DATE NOT NULL,
    cfo                         DECIMAL(24, 2) DEFAULT "0",
    depreciation                DECIMAL(24, 2) DEFAULT "0",
    cfi                         DECIMAL(24, 2) DEFAULT "0",
    capex                       DECIMAL(24, 2) DEFAULT "0",
    acquisitions                DECIMAL(24, 2) DEFAULT "0",
    cff                         DECIMAL(24, 2) DEFAULT "0",
    dividends_paid              DECIMAL(24, 2) DEFAULT "0",
    debt_issued                 DECIMAL(24, 2) DEFAULT "0",
    debt_repaid                 DECIMAL(24, 2) DEFAULT "0",
    lease_payment_interest      DECIMAL(24, 2) DEFAULT "0",
    lease_payment_principal     DECIMAL(24, 2) DEFAULT "0",
    equity_issued               DECIMAL(24, 2) DEFAULT "0",
    fcf                         DECIMAL(24, 2) DEFAULT "0",
    net_change                  DECIMAL(24, 2) DEFAULT "0",
    raw_json                    JSON,
    created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_cf_company (company_key) USING BITMAP,
    INDEX idx_cf_period (period_key) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (company_key, period_key, report_date)
PARTITION BY RANGE(report_date) () 
DISTRIBUTED BY HASH(company_key) BUCKETS 8
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

CREATE TABLE IF NOT EXISTS fact_daily_market (
    company_key             BIGINT NOT NULL,
    date                    DATE NOT NULL,
    open                    DECIMAL(18, 2) DEFAULT "0",
    high                    DECIMAL(18, 2) DEFAULT "0",
    low                     DECIMAL(18, 2) DEFAULT "0",
    close                   DECIMAL(18, 2) DEFAULT "0",
    adj_close               DECIMAL(18, 2) DEFAULT "0",
    volume                  BIGINT DEFAULT "0",
    value                   DECIMAL(24, 2) DEFAULT "0",
    market_cap              DECIMAL(26, 2) DEFAULT "0",
    shares_outstanding      DECIMAL(20, 0) DEFAULT "0",
    free_float              DECIMAL(5, 4) DEFAULT "0",
    foreign_buy             DECIMAL(24, 2) DEFAULT "0",
    foreign_sell            DECIMAL(24, 2) DEFAULT "0",
    foreign_net_buy         DECIMAL(24, 2) DEFAULT "0",
    room_left               DECIMAL(5, 4) DEFAULT "0",
    margin_ratio            DECIMAL(5, 4) DEFAULT "0",
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_market_company (company_key) USING BITMAP,
    INDEX idx_market_date (date) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (company_key, date)
PARTITION BY RANGE(date) () 
DISTRIBUTED BY HASH(company_key) BUCKETS 16
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "1",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "16"
);

CREATE TABLE IF NOT EXISTS fact_risk_metrics (
    company_key             BIGINT NOT NULL,
    date                    DATE NOT NULL,
    beta                    DECIMAL(8, 4) DEFAULT "0",
    volatility_30d          DECIMAL(8, 4) DEFAULT "0",
    volatility_90d          DECIMAL(8, 4) DEFAULT "0",
    volatility_252d         DECIMAL(8, 4) DEFAULT "0",
    return_1d               DECIMAL(10, 6) DEFAULT "0",
    return_1w               DECIMAL(10, 6) DEFAULT "0",
    return_1m               DECIMAL(10, 6) DEFAULT "0",
    return_3m               DECIMAL(10, 6) DEFAULT "0",
    return_1y               DECIMAL(10, 6) DEFAULT "0",
    volume_avg_30d          DECIMAL(20, 2) DEFAULT "0",
    volume_avg_90d          DECIMAL(20, 2) DEFAULT "0",
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_risk_company (company_key) USING BITMAP,
    INDEX idx_risk_date (date) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (company_key, date)
PARTITION BY RANGE(date) () 
DISTRIBUTED BY HASH(company_key) BUCKETS 16
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "1",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "16"
);

CREATE TABLE IF NOT EXISTS dim_macro_indicator (
    indicator_key       BIGINT NOT NULL,
    indicator_code      VARCHAR(50) NOT NULL,
    name_vn             VARCHAR(255),
    name_en             VARCHAR(255),
    unit                VARCHAR(50),
    country             VARCHAR(10),
    frequency           VARCHAR(10),
    source              VARCHAR(100),
    category            VARCHAR(50),
    is_active           TINYINT DEFAULT "1",
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_indicator_code (indicator_code) USING BITMAP
) ENGINE = OLAP
PRIMARY KEY (indicator_key)
DISTRIBUTED BY HASH(indicator_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

CREATE TABLE IF NOT EXISTS fact_macro_timeseries (
    indicator_key       BIGINT NOT NULL,
    date                DATE NOT NULL,
    value               DECIMAL(24, 4) DEFAULT "0",
    yoy                 DECIMAL(10, 4) DEFAULT "0",
    mom                 DECIMAL(10, 4) DEFAULT "0",
    qoq                 DECIMAL(10, 4) DEFAULT "0",
    note                VARCHAR(500),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_macro_indicator (indicator_key) USING BITMAP,
    INDEX idx_macro_date (date) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (indicator_key, date)
PARTITION BY RANGE(date) () 
DISTRIBUTED BY HASH(indicator_key) BUCKETS 8
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-30",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

CREATE TABLE IF NOT EXISTS fact_bond_data (
    bond_key            BIGINT NOT NULL,
    company_key         BIGINT NOT NULL,
    bond_code           VARCHAR(50),
    issuance_date       DATE,
    maturity_date       DATE,
    coupon_rate         DECIMAL(7, 4) DEFAULT "0",
    market_rate         DECIMAL(7, 4) DEFAULT "0",
    face_value          DECIMAL(24, 2) DEFAULT "0",
    current_price       DECIMAL(18, 4) DEFAULT "0",
    fair_value          DECIMAL(18, 4) DEFAULT "0",
    accrued_interest    DECIMAL(18, 4) DEFAULT "0",
    coupon_frequency    VARCHAR(20),
    yield_to_maturity   DECIMAL(10, 6) DEFAULT "0",
    duration            DECIMAL(10, 4) DEFAULT "0",
    modified_duration   DECIMAL(10, 4) DEFAULT "0",
    convexity           DECIMAL(12, 6) DEFAULT "0",
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_bond_company (company_key) USING BITMAP
) ENGINE = OLAP
PRIMARY KEY (bond_key)
DISTRIBUTED BY HASH(bond_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

CREATE TABLE IF NOT EXISTS fact_forecast (
    company_key         BIGINT NOT NULL,
    year                SMALLINT NOT NULL,
    quarter             TINYINT DEFAULT "0",
    scenario            VARCHAR(20),
    kpi                 VARCHAR(50),
    value               DECIMAL(24, 4) DEFAULT "0",
    confidence_lower    DECIMAL(24, 4) DEFAULT "0",
    confidence_upper    DECIMAL(24, 4) DEFAULT "0",
    source              VARCHAR(50),
    model_version       VARCHAR(50),
    note                VARCHAR(500),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_forecast_company (company_key) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (company_key, year, quarter, scenario, kpi)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

CREATE TABLE IF NOT EXISTS fact_budget (
    company_key         BIGINT NOT NULL,
    period_key          BIGINT NOT NULL,
    budget_revenue      DECIMAL(24, 2) DEFAULT "0",
    budget_profit       DECIMAL(24, 2) DEFAULT "0",
    budget_capex        DECIMAL(24, 2) DEFAULT "0",
    budget_opex         DECIMAL(24, 2) DEFAULT "0",
    created_by          VARCHAR(100),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (company_key, period_key)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

CREATE TABLE IF NOT EXISTS fact_sector_benchmark (
    sector              VARCHAR(100) NOT NULL,
    year                SMALLINT NOT NULL,
    quarter             TINYINT NOT NULL,
    avg_roe             DECIMAL(10, 4) SUM DEFAULT "0",
    median_roe          DECIMAL(10, 4) REPLACE DEFAULT "0",
    avg_pe              DECIMAL(10, 4) SUM DEFAULT "0",
    median_pe           DECIMAL(10, 4) REPLACE DEFAULT "0",
    avg_gross_margin    DECIMAL(10, 4) SUM DEFAULT "0",
    avg_net_margin      DECIMAL(10, 4) SUM DEFAULT "0",
    avg_debt_to_equity  DECIMAL(10, 4) SUM DEFAULT "0",
    company_count       INT SUM DEFAULT "0",
    created_at          DATETIME MAX DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
AGGREGATE KEY (sector, year, quarter)
DISTRIBUTED BY HASH(sector) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

CREATE TABLE IF NOT EXISTS mart_master_analysis (
    company_key             BIGINT NOT NULL,
    year                    SMALLINT NOT NULL,
    quarter                 TINYINT NOT NULL,
    report_date             DATE NOT NULL,
    price                   DECIMAL(18, 2) DEFAULT "0",
    pe_ttm                  DECIMAL(10, 4) DEFAULT "0",
    pb                      DECIMAL(10, 4) DEFAULT "0",
    ps                      DECIMAL(10, 4) DEFAULT "0",
    pcf                     DECIMAL(10, 4) DEFAULT "0",
    peg                     DECIMAL(10, 4) DEFAULT "0",
    ev_ebitda               DECIMAL(10, 4) DEFAULT "0",
    ev_sales                DECIMAL(10, 4) DEFAULT "0",
    market_cap_b            DECIMAL(18, 4) DEFAULT "0",
    enterprise_value        DECIMAL(26, 2) DEFAULT "0",
    dividend_yield          DECIMAL(8, 4) DEFAULT "0",
    roe_ttm                 DECIMAL(10, 4) DEFAULT "0",
    roa_ttm                 DECIMAL(10, 4) DEFAULT "0",
    roic                    DECIMAL(10, 4) DEFAULT "0",
    gross_margin            DECIMAL(10, 4) DEFAULT "0",
    operating_margin        DECIMAL(10, 4) DEFAULT "0",
    net_margin              DECIMAL(10, 4) DEFAULT "0",
    ebitda_margin           DECIMAL(10, 4) DEFAULT "0",
    asset_turnover          DECIMAL(10, 4) DEFAULT "0",
    equity_multiplier       DECIMAL(10, 4) DEFAULT "0",
    dupont_roe              DECIMAL(10, 4) DEFAULT "0",
    revenue_growth_yoy      DECIMAL(10, 4) DEFAULT "0",
    profit_growth_yoy       DECIMAL(10, 4) DEFAULT "0",
    eps_growth_yoy          DECIMAL(10, 4) DEFAULT "0",
    asset_growth_yoy        DECIMAL(10, 4) DEFAULT "0",
    equity_growth_yoy       DECIMAL(10, 4) DEFAULT "0",
    revenue_cagr_3y         DECIMAL(10, 4) DEFAULT "0",
    profit_cagr_3y          DECIMAL(10, 4) DEFAULT "0",
    eps_cagr_3y             DECIMAL(10, 4) DEFAULT "0",
    debt_to_equity          DECIMAL(10, 4) DEFAULT "0",
    debt_to_assets          DECIMAL(10, 4) DEFAULT "0",
    interest_coverage       DECIMAL(10, 4) DEFAULT "0",
    current_ratio           DECIMAL(10, 4) DEFAULT "0",
    quick_ratio             DECIMAL(10, 4) DEFAULT "0",
    cash_ratio              DECIMAL(10, 4) DEFAULT "0",
    working_capital         DECIMAL(24, 2) DEFAULT "0",
    net_debt                DECIMAL(24, 2) DEFAULT "0",
    net_debt_to_ebitda      DECIMAL(10, 4) DEFAULT "0",
    fcf_ttm                 DECIMAL(24, 2) DEFAULT "0",
    fcf_yield               DECIMAL(10, 4) DEFAULT "0",
    fcf_conversion          DECIMAL(10, 4) DEFAULT "0",
    cfo_to_revenue          DECIMAL(10, 4) DEFAULT "0",
    capex_to_revenue        DECIMAL(10, 4) DEFAULT "0",
    accrual_ratio           DECIMAL(10, 4) DEFAULT "0",
    receivables_turnover    DECIMAL(10, 4) DEFAULT "0",
    inventory_turnover      DECIMAL(10, 4) DEFAULT "0",
    payables_turnover       DECIMAL(10, 4) DEFAULT "0",
    days_receivables        DECIMAL(10, 2) DEFAULT "0",
    days_inventory          DECIMAL(10, 2) DEFAULT "0",
    days_payables           DECIMAL(10, 2) DEFAULT "0",
    cash_conversion_cycle   DECIMAL(10, 2) DEFAULT "0",
    piotroski_f_score       TINYINT DEFAULT "0",
    altman_z_score          DECIMAL(10, 4) DEFAULT "0",
    beneish_m_score         DECIMAL(10, 4) DEFAULT "0",
    sloan_ratio             DECIMAL(10, 4) DEFAULT "0",
    beta                    DECIMAL(8, 4) DEFAULT "0",
    volatility_30d          DECIMAL(8, 4) DEFAULT "0",
    volume_avg_30d          DECIMAL(20, 2) DEFAULT "0",
    foreign_ownership       DECIMAL(8, 4) DEFAULT "0",
    foreign_net_buy_ttm     DECIMAL(24, 2) DEFAULT "0",
    room_left               DECIMAL(8, 4) DEFAULT "0",
    roe_vs_sector           DECIMAL(10, 4) DEFAULT "0",
    margin_vs_sector        DECIMAL(10, 4) DEFAULT "0",
    pe_vs_sector            DECIMAL(10, 4) DEFAULT "0",
    growth_vs_sector        DECIMAL(10, 4) DEFAULT "0",
    sector_rank             SMALLINT DEFAULT "0",
    lease_adjusted_net_debt DECIMAL(24, 2) DEFAULT "0",
    tax_shield              DECIMAL(24, 2) DEFAULT "0",
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_mart_company (company_key) USING BITMAP,
    INDEX idx_mart_pe (pe_ttm) USING BITMAP,
    INDEX idx_mart_roe (roe_ttm) USING BITMAP
) ENGINE = OLAP
DUPLICATE KEY (company_key, year, quarter, report_date)
PARTITION BY RANGE(report_date) ()
DISTRIBUTED BY HASH(company_key) BUCKETS 16
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-5",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "16"
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id              BIGINT NOT NULL,
    user_id             VARCHAR(100),
    action              VARCHAR(20),
    table_name          VARCHAR(100),
    query_hash          VARCHAR(64),
    query_text          STRING,
    rows_affected       BIGINT,
    execution_time_ms   INT,
    ip_address          VARCHAR(50),
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (log_id, user_id) 
PARTITION BY RANGE(timestamp) ()
DISTRIBUTED BY HASH(log_id) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "MONTH",
    "dynamic_partition.start" = "-6",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "4"
);

CREATE TABLE IF NOT EXISTS rollup_monthly_market (
    company_key         BIGINT NOT NULL,
    year_month          DATE NOT NULL,
    avg_close           DECIMAL(18, 4) SUM,
    total_volume        BIGINT SUM,
    total_value         DECIMAL(24, 2) SUM,
    high_price          DECIMAL(18, 2) MAX,
    low_price           DECIMAL(18, 2) MIN
) ENGINE = OLAP
AGGREGATE KEY (company_key, year_month)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_market_summary
DISTRIBUTED BY HASH(date) BUCKETS 8
REFRESH ASYNC
AS
SELECT 
    date,
    COUNT(DISTINCT company_key) AS stocks_traded,
    SUM(volume) AS total_volume,
    SUM(value) AS total_value,
    AVG(close) AS avg_price,
    SUM(foreign_net_buy) AS total_foreign_net_buy
FROM fact_daily_market
GROUP BY date;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_sector_daily_performance
DISTRIBUTED BY HASH(sector) BUCKETS 8
REFRESH ASYNC
AS
SELECT 
    c.sector,
    m.date,
    COUNT(DISTINCT c.company_key) as stock_count,
    AVG(m.close) as avg_price,
    SUM(m.volume) as total_volume,
    AVG(m.foreign_net_buy) as avg_foreign_flow
FROM dim_company c
JOIN fact_daily_market m ON m.company_key = c.company_key
WHERE c.is_active = 1
GROUP BY c.sector, m.date;

CREATE USER IF NOT EXISTS 'rag_user'@'%' IDENTIFIED BY 'rag_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON analytics.* TO 'rag_user'@'%';
GRANT SELECT ON information_schema.* TO 'rag_user'@'%';