CREATE DATABASE IF NOT EXISTS analytics;
USE analytics;

-- ========================================
-- 1. DIMENSION: Companies
-- ========================================
CREATE TABLE IF NOT EXISTS dim_company (
    company_key         INT NOT NULL AUTO_INCREMENT,           -- Surrogate key for fast joins
    symbol              VARCHAR(20) NOT NULL,
    company_id          VARCHAR(36),                           -- External ID (optional)
    company_name_vn     VARCHAR(255),
    company_name_en     VARCHAR(255),
    industry_gics       VARCHAR(100),
    sector              VARCHAR(100),
    exchange            VARCHAR(20),
    listing_date        DATE,
    shares_outstanding  DECIMAL(20, 0) DEFAULT 0,              -- Integer shares
    free_float          DECIMAL(5, 4) DEFAULT 0,               -- Percentage 0.xxxx
    foreign_room        DECIMAL(5, 4) DEFAULT 0,               -- Percentage 0.xxxx
    is_active           TINYINT DEFAULT 1,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (company_key)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT",
    "enable_persistent_index" = "true",
    "compression" = "LZ4"
);

-- Create unique index on symbol for lookups
CREATE INDEX idx_company_symbol ON dim_company (symbol) USING BITMAP;

-- ========================================
-- 2. DIMENSION: Reporting Periods
-- ========================================
CREATE TABLE IF NOT EXISTS dim_period (
    period_key          INT NOT NULL AUTO_INCREMENT,           -- Surrogate key
    year                SMALLINT NOT NULL,
    quarter             TINYINT NOT NULL,
    period_type         VARCHAR(10),
    start_date          DATE,
    end_date            DATE,
    is_latest_quarter   TINYINT DEFAULT 0,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (period_key)
DISTRIBUTED BY HASH(period_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "storage_format" = "DEFAULT"
);

-- Create composite index for fast lookup
CREATE INDEX idx_period_year_quarter ON dim_period (year, quarter) USING BITMAP;

-- ========================================
-- 2.1 HELPER: Period Mapping for YoY
-- ========================================
CREATE TABLE IF NOT EXISTS dim_period_mapping (
    current_period_key  INT NOT NULL,
    prev_period_key     INT,
    year_diff           TINYINT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (current_period_key)
DISTRIBUTED BY HASH(current_period_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

-- ========================================
-- 3. FACT: Income Statement (IS)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_income_statement (
    company_key         INT NOT NULL,
    period_key          INT NOT NULL,
    report_date         DATE NOT NULL,
    
    -- Core P&L (in millions VND or reporting currency)
    revenue                         DECIMAL(24, 2) DEFAULT 0,
    cogs                            DECIMAL(24, 2) DEFAULT 0,
    gross_profit                    DECIMAL(24, 2) DEFAULT 0,
    sgna                            DECIMAL(24, 2) DEFAULT 0,
    operating_profit                DECIMAL(24, 2) DEFAULT 0,
    financial_income                DECIMAL(24, 2) DEFAULT 0,
    financial_expense               DECIMAL(24, 2) DEFAULT 0,
    interest_expense                DECIMAL(24, 2) DEFAULT 0,
    lease_interest_expense          DECIMAL(24, 2) DEFAULT 0,
    profit_before_tax               DECIMAL(24, 2) DEFAULT 0,
    tax                             DECIMAL(24, 2) DEFAULT 0,
    effective_tax_rate              DECIMAL(7, 4) DEFAULT 0,    -- 0.xxxx (e.g., 0.2500 = 25%)
    net_income                      DECIMAL(24, 2) DEFAULT 0,
    
    -- Per Share
    eps                 DECIMAL(18, 4) DEFAULT 0,               -- Earnings per share
    eps_diluted         DECIMAL(18, 4) DEFAULT 0,
    
    -- Derived
    ebitda                              DECIMAL(24, 2) DEFAULT 0,
    depreciation_amortization_total     DECIMAL(24, 2) DEFAULT 0,
    ebit                                DECIMAL(24, 2) DEFAULT 0,
    
    -- Flexible storage
    raw_json            JSON,
    
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
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
    "dynamic_partition.start" = "-5",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

CREATE INDEX idx_is_company ON fact_income_statement (company_key) USING BITMAP;
CREATE INDEX idx_is_period ON fact_income_statement (period_key) USING BITMAP;

-- ========================================
-- 4. FACT: Balance Sheet (BS)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_balance_sheet (
    company_key         INT NOT NULL,
    period_key          INT NOT NULL,
    report_date         DATE NOT NULL,
    
    -- Assets (in millions)
    cash                    DECIMAL(24, 2) DEFAULT 0,
    short_term_invest       DECIMAL(24, 2) DEFAULT 0,
    receivables             DECIMAL(24, 2) DEFAULT 0,
    inventory               DECIMAL(24, 2) DEFAULT 0,
    total_current_assets    DECIMAL(24, 2) DEFAULT 0,
    fixed_assets            DECIMAL(24, 2) DEFAULT 0,
    lease_assets            DECIMAL(24, 2) DEFAULT 0,
    intangible_assets       DECIMAL(24, 2) DEFAULT 0,
    total_assets            DECIMAL(24, 2) DEFAULT 0,
    
    -- Liabilities (in millions)
    payables                    DECIMAL(24, 2) DEFAULT 0,
    short_term_debt             DECIMAL(24, 2) DEFAULT 0,
    lease_liabilities_current   DECIMAL(24, 2) DEFAULT 0,
    long_term_debt              DECIMAL(24, 2) DEFAULT 0,
    lease_liabilities_long      DECIMAL(24, 2) DEFAULT 0,
    total_current_liab          DECIMAL(24, 2) DEFAULT 0,
    total_liabilities           DECIMAL(24, 2) DEFAULT 0,
    
    -- Equity (in millions)
    share_capital       DECIMAL(24, 2) DEFAULT 0,
    retained_earnings   DECIMAL(24, 2) DEFAULT 0,
    total_equity        DECIMAL(24, 2) DEFAULT 0,
    
    -- Book value per share
    bvps                DECIMAL(18, 4) DEFAULT 0,
    
    raw_json            JSON,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (company_key, period_key, report_date)
PARTITION BY RANGE(report_date) ()
DISTRIBUTED BY HASH(company_key) BUCKETS 8
PROPERTIES (
    "replication_num" = "1",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-5",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

CREATE INDEX idx_bs_company ON fact_balance_sheet (company_key) USING BITMAP;
CREATE INDEX idx_bs_period ON fact_balance_sheet (period_key) USING BITMAP;

-- ========================================
-- 5. FACT: Cash Flow (CF)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_cash_flow (
    company_key         INT NOT NULL,
    period_key          INT NOT NULL,
    report_date         DATE NOT NULL,
    
    -- Operating (in millions)
    cfo                 DECIMAL(24, 2) DEFAULT 0,
    depreciation        DECIMAL(24, 2) DEFAULT 0,
    
    -- Investing (in millions)
    cfi                 DECIMAL(24, 2) DEFAULT 0,
    capex               DECIMAL(24, 2) DEFAULT 0,
    acquisitions        DECIMAL(24, 2) DEFAULT 0,
    
    -- Financing (in millions)
    cff                        DECIMAL(24, 2) DEFAULT 0,
    dividends_paid             DECIMAL(24, 2) DEFAULT 0,
    debt_issued                DECIMAL(24, 2) DEFAULT 0,
    debt_repaid                DECIMAL(24, 2) DEFAULT 0,
    lease_payment_interest     DECIMAL(24, 2) DEFAULT 0,
    lease_payment_principal    DECIMAL(24, 2) DEFAULT 0,
    equity_issued              DECIMAL(24, 2) DEFAULT 0,
    
    -- Derived (in millions)
    fcf                 DECIMAL(24, 2) DEFAULT 0,
    net_change          DECIMAL(24, 2) DEFAULT 0,
    
    raw_json            JSON,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (company_key, period_key, report_date)
PARTITION BY RANGE(report_date) ()
DISTRIBUTED BY HASH(company_key) BUCKETS 8
PROPERTIES (
    "replication_num" = "1",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-5",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

CREATE INDEX idx_cf_company ON fact_cash_flow (company_key) USING BITMAP;
CREATE INDEX idx_cf_period ON fact_cash_flow (period_key) USING BITMAP;

-- ========================================
-- 6. FACT: Daily Market Data
-- ========================================
CREATE TABLE IF NOT EXISTS fact_daily_market (
    company_key         INT NOT NULL,
    date                DATE NOT NULL,
    
    -- OHLCV - Prices in stock currency
    open                DECIMAL(18, 2) DEFAULT 0,
    high                DECIMAL(18, 2) DEFAULT 0,
    low                 DECIMAL(18, 2) DEFAULT 0,
    close               DECIMAL(18, 2) DEFAULT 0,
    adj_close           DECIMAL(18, 2) DEFAULT 0,
    volume              BIGINT DEFAULT 0,                       -- Share volume
    value               DECIMAL(24, 2) DEFAULT 0,               -- Trading value in millions
    
    -- Market metrics
    market_cap          DECIMAL(26, 2) DEFAULT 0,               -- Market cap in millions
    shares_outstanding  DECIMAL(20, 0) DEFAULT 0,
    free_float          DECIMAL(5, 4) DEFAULT 0,
    
    -- Foreign trading (in millions)
    foreign_buy         DECIMAL(24, 2) DEFAULT 0,
    foreign_sell        DECIMAL(24, 2) DEFAULT 0,
    foreign_net_buy     DECIMAL(24, 2) DEFAULT 0,
    room_left           DECIMAL(5, 4) DEFAULT 0,
    
    -- Margin
    margin_ratio        DECIMAL(5, 4) DEFAULT 0,
    
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (company_key, date)
PARTITION BY RANGE(date) ()
DISTRIBUTED BY HASH(company_key) BUCKETS 16
PROPERTIES (
    "replication_num" = "1",
    "compression" = "LZ4",
    "enable_persistent_index" = "true",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "MONTH",
    "dynamic_partition.start" = "-24",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "16"
);

CREATE INDEX idx_market_company ON fact_daily_market (company_key) USING BITMAP;
CREATE INDEX idx_market_date ON fact_daily_market (date) USING BITMAP;

-- ========================================
-- 6.1 FACT: Risk Metrics (SEPARATE TABLE)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_risk_metrics (
    company_key         INT NOT NULL,
    date                DATE NOT NULL,
    
    -- Risk metrics (ratios/percentages)
    beta                DECIMAL(8, 4) DEFAULT 0,
    volatility_30d      DECIMAL(8, 4) DEFAULT 0,
    volatility_90d      DECIMAL(8, 4) DEFAULT 0,
    volatility_252d     DECIMAL(8, 4) DEFAULT 0,
    
    -- Returns (as decimals: 0.05 = 5%)
    return_1d           DECIMAL(10, 6) DEFAULT 0,
    return_1w           DECIMAL(10, 6) DEFAULT 0,
    return_1m           DECIMAL(10, 6) DEFAULT 0,
    return_3m           DECIMAL(10, 6) DEFAULT 0,
    return_1y           DECIMAL(10, 6) DEFAULT 0,
    
    -- Volume metrics
    volume_avg_30d      DECIMAL(20, 2) DEFAULT 0,
    volume_avg_90d      DECIMAL(20, 2) DEFAULT 0,
    
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (company_key, date)
PARTITION BY RANGE(date) ()
DISTRIBUTED BY HASH(company_key) BUCKETS 16
PROPERTIES (
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "MONTH",
    "dynamic_partition.start" = "-24",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "16"
);

-- ========================================
-- 7. DIMENSION: Macro Indicators
-- ========================================
CREATE TABLE IF NOT EXISTS dim_macro_indicator (
    indicator_key       INT NOT NULL AUTO_INCREMENT,
    indicator_code      VARCHAR(50) NOT NULL,
    name_vn             VARCHAR(255),
    name_en             VARCHAR(255),
    unit                VARCHAR(50),
    country             VARCHAR(10),
    frequency           VARCHAR(10),
    source              VARCHAR(100),
    category            VARCHAR(50),
    is_active           TINYINT DEFAULT 1,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (indicator_key)
DISTRIBUTED BY HASH(indicator_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

CREATE INDEX idx_indicator_code ON dim_macro_indicator (indicator_code) USING BITMAP;

-- ========================================
-- 8. FACT: Macro Time Series
-- ========================================
CREATE TABLE IF NOT EXISTS fact_macro_timeseries (
    indicator_key       INT NOT NULL,
    date                DATE NOT NULL,
    value               DECIMAL(24, 4) DEFAULT 0,               -- Flexible precision
    yoy                 DECIMAL(10, 4) DEFAULT 0,               -- YoY growth %
    mom                 DECIMAL(10, 4) DEFAULT 0,               -- MoM growth %
    qoq                 DECIMAL(10, 4) DEFAULT 0,               -- QoQ growth %
    note                VARCHAR(500),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (indicator_key, date)
PARTITION BY RANGE(date) ()
DISTRIBUTED BY HASH(indicator_key) BUCKETS 8
PROPERTIES (
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "8"
);

-- ========================================
-- 9. FACT: Bond Data
-- ========================================
CREATE TABLE IF NOT EXISTS fact_bond_data (
    bond_key            INT NOT NULL AUTO_INCREMENT,
    company_key         INT NOT NULL,
    bond_code           VARCHAR(50),
    issuance_date       DATE,
    maturity_date       DATE,
    coupon_rate         DECIMAL(7, 4) DEFAULT 0,                -- e.g., 0.0650 = 6.5%
    market_rate         DECIMAL(7, 4) DEFAULT 0,
    face_value          DECIMAL(24, 2) DEFAULT 0,
    current_price       DECIMAL(18, 4) DEFAULT 0,
    fair_value          DECIMAL(18, 4) DEFAULT 0,
    accrued_interest    DECIMAL(18, 4) DEFAULT 0,
    coupon_frequency    VARCHAR(20),
    
    -- Calculated metrics
    yield_to_maturity   DECIMAL(10, 6) DEFAULT 0,
    duration            DECIMAL(10, 4) DEFAULT 0,
    modified_duration   DECIMAL(10, 4) DEFAULT 0,
    convexity           DECIMAL(12, 6) DEFAULT 0,
    
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (bond_key)
DISTRIBUTED BY HASH(bond_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

CREATE INDEX idx_bond_company ON fact_bond_data (company_key) USING BITMAP;

-- ========================================
-- 10. FACT: Forecast
-- ========================================
CREATE TABLE IF NOT EXISTS fact_forecast (
    company_key         INT NOT NULL,
    year                SMALLINT NOT NULL,
    quarter             TINYINT DEFAULT 0,
    scenario            VARCHAR(20),
    kpi                 VARCHAR(50),
    value               DECIMAL(24, 4) DEFAULT 0,
    confidence_lower    DECIMAL(24, 4) DEFAULT 0,
    confidence_upper    DECIMAL(24, 4) DEFAULT 0,
    source              VARCHAR(50),
    model_version       VARCHAR(50),
    note                VARCHAR(500),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (company_key, year, quarter, scenario, kpi)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

-- ========================================
-- 11. FACT: Budget
-- ========================================
CREATE TABLE IF NOT EXISTS fact_budget (
    company_key         INT NOT NULL,
    period_key          INT NOT NULL,
    budget_revenue      DECIMAL(24, 2) DEFAULT 0,
    budget_profit       DECIMAL(24, 2) DEFAULT 0,
    budget_capex        DECIMAL(24, 2) DEFAULT 0,
    budget_opex         DECIMAL(24, 2) DEFAULT 0,
    created_by          VARCHAR(100),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
PRIMARY KEY (company_key, period_key)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

-- ========================================
-- 12. SECTOR BENCHMARKS (Aggregating Table)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_sector_benchmark (
    sector              VARCHAR(100) NOT NULL,
    year                SMALLINT NOT NULL,
    quarter             TINYINT NOT NULL,
    
    -- Aggregated metrics (percentages as decimals)
    avg_roe             DECIMAL(10, 4) DEFAULT 0,
    median_roe          DECIMAL(10, 4) DEFAULT 0,
    avg_pe              DECIMAL(10, 4) DEFAULT 0,
    median_pe           DECIMAL(10, 4) DEFAULT 0,
    avg_gross_margin    DECIMAL(10, 4) DEFAULT 0,
    avg_net_margin      DECIMAL(10, 4) DEFAULT 0,
    avg_debt_to_equity  DECIMAL(10, 4) DEFAULT 0,
    
    company_count       INT DEFAULT 0,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
AGGREGATE KEY (sector, year, quarter)
DISTRIBUTED BY HASH(sector) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

-- ========================================
-- 13. MASTER ANALYSIS MART
-- ========================================
CREATE TABLE IF NOT EXISTS mart_master_analysis (
    company_key         INT NOT NULL,
    year                SMALLINT NOT NULL,
    quarter             TINYINT NOT NULL,
    report_date         DATE NOT NULL,
    
    -- Price
    price               DECIMAL(18, 2) DEFAULT 0,
    
    -- === VALUATION (10 metrics) ===
    pe_ttm              DECIMAL(10, 4) DEFAULT 0,
    pb                  DECIMAL(10, 4) DEFAULT 0,
    ps                  DECIMAL(10, 4) DEFAULT 0,
    pcf                 DECIMAL(10, 4) DEFAULT 0,
    peg                 DECIMAL(10, 4) DEFAULT 0,
    ev_ebitda           DECIMAL(10, 4) DEFAULT 0,
    ev_sales            DECIMAL(10, 4) DEFAULT 0,
    market_cap_b        DECIMAL(18, 4) DEFAULT 0,                -- Market cap in billions
    enterprise_value    DECIMAL(26, 2) DEFAULT 0,
    dividend_yield      DECIMAL(8, 4) DEFAULT 0,
    
    -- === PROFITABILITY (15 metrics) - All as percentages in decimal ===
    roe_ttm             DECIMAL(10, 4) DEFAULT 0,
    roa_ttm             DECIMAL(10, 4) DEFAULT 0,
    roic                DECIMAL(10, 4) DEFAULT 0,
    gross_margin        DECIMAL(10, 4) DEFAULT 0,
    operating_margin    DECIMAL(10, 4) DEFAULT 0,
    net_margin          DECIMAL(10, 4) DEFAULT 0,
    ebitda_margin       DECIMAL(10, 4) DEFAULT 0,
    asset_turnover      DECIMAL(10, 4) DEFAULT 0,
    equity_multiplier   DECIMAL(10, 4) DEFAULT 0,
    dupont_roe          DECIMAL(10, 4) DEFAULT 0,
    
    -- === GROWTH (10 metrics) ===
    revenue_growth_yoy  DECIMAL(10, 4) DEFAULT 0,
    profit_growth_yoy   DECIMAL(10, 4) DEFAULT 0,
    eps_growth_yoy      DECIMAL(10, 4) DEFAULT 0,
    asset_growth_yoy    DECIMAL(10, 4) DEFAULT 0,
    equity_growth_yoy   DECIMAL(10, 4) DEFAULT 0,
    revenue_cagr_3y     DECIMAL(10, 4) DEFAULT 0,
    profit_cagr_3y      DECIMAL(10, 4) DEFAULT 0,
    eps_cagr_3y         DECIMAL(10, 4) DEFAULT 0,
    
    -- === LEVERAGE & LIQUIDITY (12 metrics) ===
    debt_to_equity      DECIMAL(10, 4) DEFAULT 0,
    debt_to_assets      DECIMAL(10, 4) DEFAULT 0,
    interest_coverage   DECIMAL(10, 4) DEFAULT 0,
    current_ratio       DECIMAL(10, 4) DEFAULT 0,
    quick_ratio         DECIMAL(10, 4) DEFAULT 0,
    cash_ratio          DECIMAL(10, 4) DEFAULT 0,
    working_capital     DECIMAL(24, 2) DEFAULT 0,
    net_debt            DECIMAL(24, 2) DEFAULT 0,
    net_debt_to_ebitda  DECIMAL(10, 4) DEFAULT 0,
    
    -- === CASH FLOW QUALITY (8 metrics) ===
    fcf_ttm             DECIMAL(24, 2) DEFAULT 0,
    fcf_yield           DECIMAL(10, 4) DEFAULT 0,
    fcf_conversion      DECIMAL(10, 4) DEFAULT 0,
    cfo_to_revenue      DECIMAL(10, 4) DEFAULT 0,
    capex_to_revenue    DECIMAL(10, 4) DEFAULT 0,
    accrual_ratio       DECIMAL(10, 4) DEFAULT 0,
    
    -- === EFFICIENCY (8 metrics) ===
    receivables_turnover DECIMAL(10, 4) DEFAULT 0,
    inventory_turnover  DECIMAL(10, 4) DEFAULT 0,
    payables_turnover   DECIMAL(10, 4) DEFAULT 0,
    days_receivables    DECIMAL(10, 2) DEFAULT 0,
    days_inventory      DECIMAL(10, 2) DEFAULT 0,
    days_payables       DECIMAL(10, 2) DEFAULT 0,
    cash_conversion_cycle DECIMAL(10, 2) DEFAULT 0,
    
    -- === QUALITY SCORES (5 metrics) ===
    piotroski_f_score   TINYINT DEFAULT 0,
    altman_z_score      DECIMAL(10, 4) DEFAULT 0,
    beneish_m_score     DECIMAL(10, 4) DEFAULT 0,
    sloan_ratio         DECIMAL(10, 4) DEFAULT 0,
    
    -- === MARKET DATA ===
    beta                DECIMAL(8, 4) DEFAULT 0,
    volatility_30d      DECIMAL(8, 4) DEFAULT 0,
    volume_avg_30d      DECIMAL(20, 2) DEFAULT 0,
    
    -- === FOREIGN DATA ===
    foreign_ownership   DECIMAL(8, 4) DEFAULT 0,
    foreign_net_buy_ttm DECIMAL(24, 2) DEFAULT 0,
    room_left           DECIMAL(8, 4) DEFAULT 0,
    
    -- === SECTOR COMPARISON (5 metrics) ===
    roe_vs_sector       DECIMAL(10, 4) DEFAULT 0,
    margin_vs_sector    DECIMAL(10, 4) DEFAULT 0,
    pe_vs_sector        DECIMAL(10, 4) DEFAULT 0,
    growth_vs_sector    DECIMAL(10, 4) DEFAULT 0,
    sector_rank         SMALLINT DEFAULT 0,
    
    lease_adjusted_net_debt DECIMAL(24, 2) DEFAULT 0,
    tax_shield          DECIMAL(24, 2) DEFAULT 0,
    
    -- Metadata
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE = OLAP
DUPLICATE KEY (company_key, year, quarter, report_date)
PARTITION BY RANGE(report_date) ()
DISTRIBUTED BY HASH(company_key) BUCKETS 16
PROPERTIES (
    "replication_num" = "1",
    "compression" = "LZ4",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "YEAR",
    "dynamic_partition.start" = "-5",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "16"
);

CREATE INDEX idx_mart_pe ON mart_master_analysis (pe_ttm) USING BITMAP;
CREATE INDEX idx_mart_roe ON mart_master_analysis (roe_ttm) USING BITMAP;
CREATE INDEX idx_mart_company ON mart_master_analysis (company_key) USING BITMAP;

-- ========================================
-- AUDIT LOG TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS audit_log (
    log_id              BIGINT NOT NULL AUTO_INCREMENT,
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
PRIMARY KEY (log_id)
PARTITION BY RANGE(timestamp) ()
DISTRIBUTED BY HASH(log_id) BUCKETS 4
PROPERTIES (
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "MONTH",
    "dynamic_partition.start" = "-6",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "4"
);

-- ========================================
-- ROLLUP TABLES FOR AGGREGATIONS
-- ========================================
CREATE TABLE IF NOT EXISTS rollup_monthly_market (
    company_key         INT NOT NULL,
    year_month          DATE NOT NULL,
    avg_close           DECIMAL(18, 4),
    total_volume        BIGINT,
    total_value         DECIMAL(24, 2),
    high_price          DECIMAL(18, 2),
    low_price           DECIMAL(18, 2)
) ENGINE = OLAP
AGGREGATE KEY (company_key, year_month)
DISTRIBUTED BY HASH(company_key) BUCKETS 4
PROPERTIES (
    "replication_num" = "1"
);

-- ========================================
-- PRODUCTION-GRADE FINANCIAL ANALYSIS SYSTEM
-- StarRocks OLAP for Financial Statements + Macro + Stock + Screener + AI RAG
-- Version: 2.2 (StarRocks - Optimized with DECIMAL + Surrogate Keys)
-- COMPLETION
-- ========================================

-- Continuing from dim_company sample data insertion...

INSERT INTO dim_company (symbol, company_id, company_name_vn, company_name_en, 
                         industry_gics, sector, exchange, listing_date, 
                         shares_outstanding, free_float, foreign_room, is_active) VALUES
('HPG', UUID(), 'Tập đoàn Hòa Phát', 'Hoa Phat Group', 
 'Steel', 'Materials', 'HOSE', '2007-11-16', 5000000000, 0.85, 0.45, 1),
('VCB', UUID(), 'Ngân hàng TMCP Ngoại thương Việt Nam', 'Vietcombank',
 'Banking', 'Financials', 'HOSE', '2009-07-14', 3709087510, 0.65, 0.00, 1),
('VNM', UUID(), 'Công ty Cổ phần Sữa Việt Nam', 'Vietnam Dairy Products',
 'Food Products', 'Consumer Staples', 'HOSE', '2006-01-19', 1746214263, 0.55, 0.49, 1),
('VIC', UUID(), 'Tập đoàn Vingroup', 'Vingroup JSC',
 'Real Estate', 'Real Estate', 'HOSE', '2007-07-19', 4284312500, 0.70, 0.49, 1),
('FPT', UUID(), 'Công ty Cổ phần FPT', 'FPT Corporation',
 'IT Services', 'Information Technology', 'HOSE', '2006-12-08', 1177762020, 0.80, 0.49, 1);

-- Insert sample periods
INSERT INTO dim_period (year, quarter, period_type, start_date, end_date, is_latest_quarter) VALUES
(2023, 1, 'Q', '2023-01-01', '2023-03-31', 0),
(2023, 2, 'Q', '2023-04-01', '2023-06-30', 0),
(2023, 3, 'Q', '2023-07-01', '2023-09-30', 0),
(2023, 4, 'Q', '2023-10-01', '2023-12-31', 0),
(2024, 1, 'Q', '2024-01-01', '2024-03-31', 0),
(2024, 2, 'Q', '2024-04-01', '2024-06-30', 0),
(2024, 3, 'Q', '2024-07-01', '2024-09-30', 1),
(2024, 4, 'Q', '2024-10-01', '2024-12-31', 0);

-- Insert period mappings for YoY comparisons
INSERT INTO dim_period_mapping (current_period_key, prev_period_key, year_diff)
SELECT 
    curr.period_key,
    prev.period_key,
    1 as year_diff
FROM dim_period curr
LEFT JOIN dim_period prev 
    ON curr.quarter = prev.quarter 
    AND curr.year = prev.year + 1;

-- Insert sample macro indicators
INSERT INTO dim_macro_indicator (indicator_code, name_vn, name_en, unit, country, 
                                 frequency, source, category, is_active) VALUES
('GDP_GROWTH', 'Tăng trưởng GDP', 'GDP Growth Rate', '%', 'VN', 'Quarterly', 'GSO', 'Economic', 1),
('CPI', 'Chỉ số giá tiêu dùng', 'Consumer Price Index', 'Index', 'VN', 'Monthly', 'GSO', 'Inflation', 1),
('INTEREST_RATE', 'Lãi suất cơ bản', 'Base Interest Rate', '%', 'VN', 'Monthly', 'SBV', 'Monetary', 1),
('USD_VND', 'Tỷ giá USD/VND', 'USD/VND Exchange Rate', 'VND', 'VN', 'Daily', 'SBV', 'FX', 1),
('PMI', 'Chỉ số PMI', 'Purchasing Managers Index', 'Index', 'VN', 'Monthly', 'IHS', 'Economic', 1),
('FDI', 'Vốn FDI đăng ký', 'Registered FDI', 'USD M', 'VN', 'Monthly', 'MPI', 'Investment', 1),
('EXPORT', 'Kim ngạch xuất khẩu', 'Export Value', 'USD B', 'VN', 'Monthly', 'GSO', 'Trade', 1),
('IMPORT', 'Kim ngạch nhập khẩu', 'Import Value', 'USD B', 'VN', 'Monthly', 'GSO', 'Trade', 1);

-- ========================================
-- ANALYTICAL VIEWS
-- ========================================

-- View: Latest Financial Snapshot
CREATE VIEW IF NOT EXISTS v_latest_financials AS
SELECT 
    c.company_key,
    c.symbol,
    c.company_name_en,
    c.sector,
    p.year,
    p.quarter,
    i.revenue,
    i.net_income,
    i.eps,
    i.ebitda,
    b.total_assets,
    b.total_equity,
    b.total_liabilities,
    cf.fcf,
    cf.cfo
FROM dim_company c
JOIN dim_period p ON p.is_latest_quarter = 1
LEFT JOIN fact_income_statement i ON i.company_key = c.company_key AND i.period_key = p.period_key
LEFT JOIN fact_balance_sheet b ON b.company_key = c.company_key AND b.period_key = p.period_key
LEFT JOIN fact_cash_flow cf ON cf.company_key = c.company_key AND cf.period_key = p.period_key
WHERE c.is_active = 1;

-- View: TTM Metrics (Trailing Twelve Months)
CREATE VIEW IF NOT EXISTS v_ttm_metrics AS
SELECT 
    c.company_key,
    c.symbol,
    SUM(i.revenue) as revenue_ttm,
    SUM(i.net_income) as net_income_ttm,
    SUM(i.operating_profit) as operating_profit_ttm,
    SUM(i.ebitda) as ebitda_ttm,
    SUM(cf.fcf) as fcf_ttm,
    SUM(cf.cfo) as cfo_ttm,
    SUM(cf.capex) as capex_ttm
FROM dim_company c
JOIN fact_income_statement i ON i.company_key = c.company_key
JOIN fact_cash_flow cf ON cf.company_key = c.company_key AND cf.period_key = i.period_key
WHERE i.report_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY c.company_key, c.symbol;

-- View: YoY Growth Analysis
CREATE VIEW IF NOT EXISTS v_yoy_growth AS
SELECT 
    c.symbol,
    c.company_name_en,
    curr_p.year as current_year,
    curr_p.quarter as current_quarter,
    curr_i.revenue as current_revenue,
    prev_i.revenue as prev_revenue,
    ROUND((curr_i.revenue - prev_i.revenue) / NULLIF(prev_i.revenue, 0) * 100, 2) as revenue_growth_pct,
    curr_i.net_income as current_net_income,
    prev_i.net_income as prev_net_income,
    ROUND((curr_i.net_income - prev_i.net_income) / NULLIF(prev_i.net_income, 0) * 100, 2) as profit_growth_pct
FROM dim_company c
JOIN fact_income_statement curr_i ON curr_i.company_key = c.company_key
JOIN dim_period curr_p ON curr_p.period_key = curr_i.period_key
JOIN dim_period_mapping pm ON pm.current_period_key = curr_p.period_key
JOIN fact_income_statement prev_i ON prev_i.company_key = c.company_key 
    AND prev_i.period_key = pm.prev_period_key
WHERE c.is_active = 1;

-- View: Valuation Multiples
CREATE VIEW IF NOT EXISTS v_valuation_multiples AS
SELECT 
    c.symbol,
    c.company_name_en,
    m.date,
    m.close as price,
    m.market_cap,
    ttm.net_income_ttm,
    ttm.revenue_ttm,
    ttm.ebitda_ttm,
    ttm.fcf_ttm,
    b.total_equity,
    ROUND(m.market_cap / NULLIF(ttm.net_income_ttm, 0), 2) as pe_ttm,
    ROUND(m.market_cap / NULLIF(b.total_equity, 0), 2) as pb,
    ROUND(m.market_cap / NULLIF(ttm.revenue_ttm, 0), 2) as ps,
    ROUND((m.market_cap + b.total_liabilities - b.cash) / NULLIF(ttm.ebitda_ttm, 0), 2) as ev_ebitda
FROM dim_company c
JOIN fact_daily_market m ON m.company_key = c.company_key
    AND m.date = (SELECT MAX(date) FROM fact_daily_market WHERE company_key = c.company_key)
LEFT JOIN v_ttm_metrics ttm ON ttm.company_key = c.company_key
LEFT JOIN fact_balance_sheet b ON b.company_key = c.company_key
    AND b.period_key = (SELECT MAX(period_key) FROM fact_balance_sheet WHERE company_key = c.company_key)
WHERE c.is_active = 1;

-- View: Quality & Risk Scores
CREATE VIEW IF NOT EXISTS v_quality_scores AS
SELECT 
    c.symbol,
    c.company_name_en,
    m.piotroski_f_score,
    m.altman_z_score,
    m.beneish_m_score,
    m.beta,
    m.volatility_30d,
    r.return_1y,
    CASE 
        WHEN m.piotroski_f_score >= 7 THEN 'Strong'
        WHEN m.piotroski_f_score >= 5 THEN 'Moderate'
        ELSE 'Weak'
    END as quality_rating,
    CASE 
        WHEN m.altman_z_score > 3 THEN 'Safe'
        WHEN m.altman_z_score > 1.8 THEN 'Grey Zone'
        ELSE 'Distress'
    END as financial_health
FROM dim_company c
JOIN mart_master_analysis m ON m.company_key = c.company_key
    AND m.report_date = (SELECT MAX(report_date) FROM mart_master_analysis WHERE company_key = c.company_key)
LEFT JOIN fact_risk_metrics r ON r.company_key = c.company_key
    AND r.date = (SELECT MAX(date) FROM fact_risk_metrics WHERE company_key = c.company_key)
WHERE c.is_active = 1;

-- View: Sector Performance
CREATE VIEW IF NOT EXISTS v_sector_performance AS
SELECT 
    c.sector,
    COUNT(DISTINCT c.company_key) as company_count,
    ROUND(AVG(m.roe_ttm), 4) as avg_roe,
    ROUND(AVG(m.pe_ttm), 2) as avg_pe,
    ROUND(AVG(m.net_margin), 4) as avg_net_margin,
    ROUND(AVG(m.debt_to_equity), 4) as avg_debt_to_equity,
    ROUND(AVG(m.revenue_growth_yoy), 4) as avg_revenue_growth
FROM dim_company c
JOIN mart_master_analysis m ON m.company_key = c.company_key
    AND m.report_date = (SELECT MAX(report_date) FROM mart_master_analysis WHERE company_key = c.company_key)
WHERE c.is_active = 1
GROUP BY c.sector
ORDER BY avg_roe DESC;

-- ========================================
-- USEFUL ANALYTICAL QUERIES
-- ========================================

-- Query 1: Top 10 Companies by ROE
-- SELECT symbol, company_name_en, roe_ttm, pe_ttm, net_margin
-- FROM mart_master_analysis m
-- JOIN dim_company c ON c.company_key = m.company_key
-- WHERE m.report_date = (SELECT MAX(report_date) FROM mart_master_analysis)
--   AND m.roe_ttm > 0
-- ORDER BY m.roe_ttm DESC
-- LIMIT 10;

-- Query 2: Undervalued Growth Stocks (PEG < 1, Growth > 15%)
-- SELECT c.symbol, c.company_name_en, m.pe_ttm, m.eps_growth_yoy, m.peg
-- FROM mart_master_analysis m
-- JOIN dim_company c ON c.company_key = m.company_key
-- WHERE m.report_date = (SELECT MAX(report_date) FROM mart_master_analysis)
--   AND m.peg < 1 AND m.peg > 0
--   AND m.eps_growth_yoy > 0.15
-- ORDER BY m.peg ASC;

-- Query 3: High Quality Dividend Stocks
-- SELECT c.symbol, c.company_name_en, m.dividend_yield, m.piotroski_f_score,
--        m.fcf_yield, m.debt_to_equity
-- FROM mart_master_analysis m
-- JOIN dim_company c ON c.company_key = m.company_key
-- WHERE m.report_date = (SELECT MAX(report_date) FROM mart_master_analysis)
--   AND m.dividend_yield > 0.03
--   AND m.piotroski_f_score >= 7
--   AND m.fcf_yield > 0.05
--   AND m.debt_to_equity < 1
-- ORDER BY m.dividend_yield DESC;

-- Query 4: Companies with Improving Cash Flow
-- SELECT c.symbol, c.company_name_en, 
--        curr.fcf as current_fcf, prev.fcf as prev_fcf,
--        ROUND((curr.fcf - prev.fcf) / NULLIF(ABS(prev.fcf), 0) * 100, 2) as fcf_growth
-- FROM dim_company c
-- JOIN fact_cash_flow curr ON curr.company_key = c.company_key
-- JOIN dim_period curr_p ON curr_p.period_key = curr.period_key AND curr_p.is_latest_quarter = 1
-- JOIN dim_period_mapping pm ON pm.current_period_key = curr_p.period_key
-- JOIN fact_cash_flow prev ON prev.company_key = c.company_key AND prev.period_key = pm.prev_period_key
-- WHERE curr.fcf > prev.fcf AND prev.fcf > 0
-- ORDER BY fcf_growth DESC;

-- Query 5: Macro Indicators Latest Values
-- SELECT m.name_en, m.unit, t.date, t.value, t.yoy
-- FROM dim_macro_indicator m
-- JOIN fact_macro_timeseries t ON t.indicator_key = m.indicator_key
-- WHERE t.date = (SELECT MAX(date) FROM fact_macro_timeseries WHERE indicator_key = m.indicator_key)
--   AND m.is_active = 1
-- ORDER BY m.category, m.name_en;

-- ========================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ========================================

-- Materialized View: Daily Market Summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_market_summary
REFRESH ASYNC
AS
SELECT 
    date,
    COUNT(DISTINCT company_key) as stocks_traded,
    SUM(volume) as total_volume,
    SUM(value) as total_value,
    AVG(close) as avg_price,
    SUM(foreign_net_buy) as total_foreign_net_buy
FROM fact_daily_market
GROUP BY date;

-- Materialized View: Sector Daily Performance
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_sector_daily_performance
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

-- ========================================
-- INDEXES FOR COMMON QUERY PATTERNS
-- ========================================

-- Additional composite indexes for joins
CREATE INDEX idx_is_company_period ON fact_income_statement (company_key, period_key) USING BITMAP;
CREATE INDEX idx_bs_company_period ON fact_balance_sheet (company_key, period_key) USING BITMAP;
CREATE INDEX idx_cf_company_period ON fact_cash_flow (company_key, period_key) USING BITMAP;

-- Indexes for date-range queries
CREATE INDEX idx_market_company_date ON fact_daily_market (company_key, date) USING BITMAP;
CREATE INDEX idx_macro_indicator_date ON fact_macro_timeseries (indicator_key, date) USING BITMAP;

-- ========================================
-- REFRESH PROCEDURES (PSEUDO-CODE)
-- ========================================

-- Procedure to refresh mart_master_analysis
-- This would typically be scheduled to run after financial data loads
-- CALL refresh_mart_master_analysis(company_key, period_key);

-- Procedure to update sector benchmarks
-- CALL refresh_sector_benchmarks(year, quarter);

-- Procedure to calculate risk metrics
-- CALL calculate_risk_metrics(date);

-- ========================================
-- DATA RETENTION POLICY
-- ========================================

-- Daily market data: Keep 2 years in hot storage, archive older
-- Financial statements: Keep all (5+ years)
-- Audit logs: Keep 6 months
-- Macro data: Keep 10 years

-- ========================================
-- BACKUP & RECOVERY NOTES
-- ========================================

-- Daily incremental backups of fact tables
-- Weekly full backups of dimension tables
-- Monthly archives of mart tables
-- Disaster recovery RPO: 24 hours, RTO: 4 hours

-- ========================================
-- MONITORING QUERIES
-- ========================================

-- Check data freshness
-- SELECT 
--     'daily_market' as table_name,
--     MAX(date) as latest_date,
--     DATEDIFF(CURDATE(), MAX(date)) as days_behind
-- FROM fact_daily_market;

-- Check row counts
-- SELECT 
--     'fact_income_statement' as table_name, COUNT(*) as row_count FROM fact_income_statement
-- UNION ALL
-- SELECT 'fact_balance_sheet', COUNT(*) FROM fact_balance_sheet
-- UNION ALL
-- SELECT 'fact_cash_flow', COUNT(*) FROM fact_cash_flow
-- UNION ALL
-- SELECT 'fact_daily_market', COUNT(*) FROM fact_daily_market;

-- ========================================
-- END OF SCHEMA
-- ========================================

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON analytics.* TO 'analyst'@'%';
-- GRANT SELECT, INSERT, UPDATE ON analytics.* TO 'etl_user'@'%';
-- GRANT ALL PRIVILEGES ON analytics.* TO 'admin'@'%';

-- ========================================
-- SECURITY SETUP (Optional but Recommended)
-- ========================================

-- 1. Tạo user riêng cho ứng dụng RAG (pass là 'rag_password')
CREATE USER IF NOT EXISTS 'rag_user'@'%' IDENTIFIED BY 'rag_password';

-- 2. Cấp quyền SELECT, INSERT, UPDATE, DELETE cho user này trên DB analytics
GRANT SELECT, INSERT, UPDATE, DELETE ON analytics.* TO 'rag_user'@'%';

-- 3. Cấp quyền READ cho user này (nếu cần query metadata)
GRANT SELECT ON information_schema.* TO 'rag_user'@'%';

-- Flush không bắt buộc ở các bản mới nhưng cứ thêm cho chắc
FLUSH PRIVILEGES;