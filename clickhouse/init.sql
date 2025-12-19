-- ========================================
-- PRODUCTION-GRADE FINANCIAL ANALYSIS SYSTEM
-- ClickHouse OLAP for Financial Statements + Macro + Stock + Screener + AI RAG
-- Version: 2.0 (Refined)
-- ========================================

CREATE DATABASE IF NOT EXISTS analytics;
USE analytics;

-- ========================================
-- 1. DIMENSION: Companies
-- ========================================
CREATE TABLE IF NOT EXISTS dim_company (
    symbol              LowCardinality(String),
    company_id          UUID DEFAULT generateUUIDv7(),
    company_name_vn     String,
    company_name_en     String,
    industry_gics       LowCardinality(String),      -- GICS Level 4
    sector              LowCardinality(String),      -- Ngân hàng, BĐS, etc.
    exchange            Enum8('HOSE'=1,'HNX'=2,'UPCOM'=3),
    listing_date        Date,
    shares_outstanding  Float64 DEFAULT 0,
    free_float          Float32 DEFAULT 0,
    foreign_room        Float32 DEFAULT 0,           -- % room còn lại
    is_active           UInt8 DEFAULT 1,
    created_at          DateTime DEFAULT now(),
    updated_at          DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY symbol
SETTINGS index_granularity = 8192;

-- Add secondary index for fast symbol lookup
ALTER TABLE dim_company ADD INDEX idx_symbol symbol TYPE set(1000) GRANULARITY 1;

-- ========================================
-- 2. DIMENSION: Reporting Periods
-- ========================================
CREATE TABLE IF NOT EXISTS dim_period (
    period_id           UUID DEFAULT generateUUIDv7(),
    year                UInt16,
    quarter             UInt8,                       -- 1-4, 0=full year
    period_type         Enum8('Q'=1, 'YTD'=2, 'Y'=3, 'TTM'=4),
    start_date          Date,
    end_date            Date,
    is_latest_quarter   UInt8 DEFAULT 0,
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
ORDER BY (year, quarter, period_type)
SETTINGS index_granularity = 8192;

-- ========================================
-- 2.1 HELPER: Period Mapping for YoY
-- ========================================
CREATE TABLE IF NOT EXISTS dim_period_mapping (
    current_period_id   UUID,
    prev_period_id      UUID,
    year_diff           Int8,
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
ORDER BY current_period_id
SETTINGS index_granularity = 8192;

-- Populate period mapping (run after dim_period is populated)
-- INSERT INTO dim_period_mapping
-- SELECT 
--     p1.period_id AS current_period_id,
--     p2.period_id AS prev_period_id,
--     1 AS year_diff
-- FROM dim_period p1
-- LEFT JOIN dim_period p2 
--     ON p1.year = p2.year + 1 
--    AND p1.quarter = p2.quarter
--    AND p1.period_type = p2.period_type;

-- ========================================
-- 3. FACT: Income Statement (IS)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_income_statement (
    symbol              LowCardinality(String),
    period_id           UUID,
    report_date         Date,
    
    -- Core P&L
    revenue             Float64 DEFAULT 0,
    cogs                Float64 DEFAULT 0,
    gross_profit        Float64 DEFAULT 0,
    sgna                Float64 DEFAULT 0,           -- Selling, General & Admin
    operating_profit    Float64 DEFAULT 0,
    financial_income    Float64 DEFAULT 0,
    financial_expense   Float64 DEFAULT 0,
    interest_expense    Float64 DEFAULT 0,
    profit_before_tax   Float64 DEFAULT 0,
    tax                 Float64 DEFAULT 0,
    net_income          Float64 DEFAULT 0,           -- LNST công ty mẹ
    
    -- Per Share
    eps                 Float32 DEFAULT 0,
    eps_diluted         Float32 DEFAULT 0,
    
    -- Derived
    ebitda              Float64 DEFAULT 0,
    ebit                Float64 DEFAULT 0,
    
    -- Flexible storage
    extra_items         Map(LowCardinality(String), Float64),
    raw_json            String CODEC(ZSTD(3)),       -- Compressed JSON backup
    
    created_at          DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(report_date)
PARTITION BY toYYYYMM(report_date)
ORDER BY (symbol, period_id, report_date)
SETTINGS index_granularity = 8192;

-- Add compression for numeric columns
ALTER TABLE fact_income_statement MODIFY COLUMN revenue CODEC(ZSTD(1));
ALTER TABLE fact_income_statement MODIFY COLUMN net_income CODEC(ZSTD(1));

-- ========================================
-- 4. FACT: Balance Sheet (BS)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_balance_sheet (
    symbol              LowCardinality(String),
    period_id           UUID,
    report_date         Date,
    
    -- Assets
    cash                Float64 DEFAULT 0,
    short_term_invest   Float64 DEFAULT 0,
    receivables         Float64 DEFAULT 0,
    inventory           Float64 DEFAULT 0,
    total_current_assets Float64 DEFAULT 0,
    fixed_assets        Float64 DEFAULT 0,
    intangible_assets   Float64 DEFAULT 0,
    total_assets        Float64 DEFAULT 0,
    
    -- Liabilities
    payables            Float64 DEFAULT 0,
    short_term_debt     Float64 DEFAULT 0,
    long_term_debt      Float64 DEFAULT 0,
    total_current_liab  Float64 DEFAULT 0,
    total_liabilities   Float64 DEFAULT 0,
    
    -- Equity
    share_capital       Float64 DEFAULT 0,
    retained_earnings   Float64 DEFAULT 0,
    total_equity        Float64 DEFAULT 0,
    
    -- Book value
    bvps                Float32 DEFAULT 0,           -- Book value per share
    
    extra_items         Map(LowCardinality(String), Float64),
    raw_json            String CODEC(ZSTD(3)),
    created_at          DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(report_date)
PARTITION BY toYYYYMM(report_date)
ORDER BY (symbol, period_id, report_date)
SETTINGS index_granularity = 8192;

ALTER TABLE fact_balance_sheet MODIFY COLUMN total_assets CODEC(ZSTD(1));
ALTER TABLE fact_balance_sheet MODIFY COLUMN total_equity CODEC(ZSTD(1));

-- ========================================
-- 5. FACT: Cash Flow (CF)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_cash_flow (
    symbol              LowCardinality(String),
    period_id           UUID,
    report_date         Date,
    
    -- Operating
    cfo                 Float64 DEFAULT 0,           -- Cash from operations
    depreciation        Float64 DEFAULT 0,
    
    -- Investing
    cfi                 Float64 DEFAULT 0,           -- Cash from investing
    capex               Float64 DEFAULT 0,
    acquisitions        Float64 DEFAULT 0,
    
    -- Financing
    cff                 Float64 DEFAULT 0,           -- Cash from financing
    dividends_paid      Float64 DEFAULT 0,
    debt_issued         Float64 DEFAULT 0,
    debt_repaid         Float64 DEFAULT 0,
    equity_issued       Float64 DEFAULT 0,
    
    -- Derived
    fcf                 Float64 DEFAULT 0,           -- Free cash flow = CFO - Capex
    net_change          Float64 DEFAULT 0,
    
    extra_items         Map(LowCardinality(String), Float64),
    raw_json            String CODEC(ZSTD(3)),
    created_at          DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(report_date)
PARTITION BY toYYYYMM(report_date)
ORDER BY (symbol, period_id, report_date)
SETTINGS index_granularity = 8192;

ALTER TABLE fact_cash_flow MODIFY COLUMN cfo CODEC(ZSTD(1));
ALTER TABLE fact_cash_flow MODIFY COLUMN fcf CODEC(ZSTD(1));

-- ========================================
-- 6. FACT: Daily Market Data
-- ========================================
CREATE TABLE IF NOT EXISTS fact_daily_market (
    symbol              LowCardinality(String),
    date                Date,
    
    -- OHLCV
    open                Float32 DEFAULT 0,
    high                Float32 DEFAULT 0,
    low                 Float32 DEFAULT 0,
    close               Float32 DEFAULT 0,
    adj_close           Float32 DEFAULT 0,
    volume              UInt64 DEFAULT 0,
    value               Float64 DEFAULT 0,           -- Giá trị GD (tỷ VND)
    
    -- Market metrics (stored at point in time)
    market_cap          Float64 DEFAULT 0,
    shares_outstanding  Float64 DEFAULT 0,           -- ADDED: for historical accuracy
    free_float          Float32 DEFAULT 0,           -- ADDED
    
    -- Foreign trading
    foreign_buy         Float64 DEFAULT 0,
    foreign_sell        Float64 DEFAULT 0,
    foreign_net_buy     Float64 DEFAULT 0,
    room_left           Float32 DEFAULT 0,           -- % room còn lại
    
    -- Margin
    margin_ratio        Float32 DEFAULT 0,           -- Dư nợ / Room margin
    
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
PARTITION BY toYYYYMMDD(date)                        -- CHANGED: daily partition
ORDER BY (symbol, date)
SETTINGS index_granularity = 4096;                   -- CHANGED: smaller for realtime

-- Compression for high-volume columns
ALTER TABLE fact_daily_market MODIFY COLUMN close CODEC(ZSTD(1));
ALTER TABLE fact_daily_market MODIFY COLUMN volume CODEC(ZSTD(3));
ALTER TABLE fact_daily_market MODIFY COLUMN market_cap CODEC(ZSTD(3));

-- Add secondary index
ALTER TABLE fact_daily_market ADD INDEX idx_symbol_date symbol TYPE set(1000) GRANULARITY 1;

-- ========================================
-- 6.1 FACT: Risk Metrics (SEPARATE TABLE)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_risk_metrics (
    symbol              LowCardinality(String),
    date                Date,
    
    -- Risk metrics
    beta                Float32 DEFAULT 0,
    volatility_30d      Float32 DEFAULT 0,
    volatility_90d      Float32 DEFAULT 0,
    volatility_252d     Float32 DEFAULT 0,
    
    -- Returns
    return_1d           Float32 DEFAULT 0,
    return_1w           Float32 DEFAULT 0,
    return_1m           Float32 DEFAULT 0,
    return_3m           Float32 DEFAULT 0,
    return_1y           Float32 DEFAULT 0,
    
    -- Volume metrics
    volume_avg_30d      Float64 DEFAULT 0,
    volume_avg_90d      Float64 DEFAULT 0,
    
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, date)
SETTINGS index_granularity = 8192;

-- ========================================
-- 7. DIMENSION: Macro Indicators
-- ========================================
CREATE TABLE IF NOT EXISTS dim_macro_indicator (
    indicator_code      LowCardinality(String),      -- VN_CPI_YOY, SBV_RATE, etc.
    name_vn             String,
    name_en             String,
    unit                String,
    country             LowCardinality(String),      -- VN, US, CN, etc.
    frequency           Enum8('D'=1,'W'=2,'M'=3,'Q'=4,'Y'=5),
    source              String,
    category            LowCardinality(String),      -- Inflation, GDP, Credit, etc.
    is_active           UInt8 DEFAULT 1,
    created_at          DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY indicator_code
SETTINGS index_granularity = 8192;

-- ========================================
-- 8. FACT: Macro Time Series
-- ========================================
CREATE TABLE IF NOT EXISTS fact_macro_timeseries (
    indicator_code      LowCardinality(String),
    date                Date,
    value               Float64 DEFAULT 0,
    yoy                 Float32 DEFAULT 0,           -- Year-over-year %
    mom                 Float32 DEFAULT 0,           -- Month-over-month %
    qoq                 Float32 DEFAULT 0,           -- Quarter-over-quarter %
    note                String,
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
PARTITION BY toYYYYMM(date)
ORDER BY (indicator_code, date)
SETTINGS index_granularity = 8192;

-- ========================================
-- 9. FACT: Bond Data (NEW)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_bond_data (
    symbol              LowCardinality(String),
    bond_id             UUID DEFAULT generateUUIDv7(),
    bond_code           String,
    issuance_date       Date,
    maturity_date       Date,
    coupon_rate         Float32 DEFAULT 0,
    face_value          Float64 DEFAULT 0,
    current_price       Float64 DEFAULT 0,
    coupon_frequency    Enum8('Annual'=1,'SemiAnnual'=2,'Quarterly'=3),
    
    -- Calculated metrics
    yield_to_maturity   Float64 DEFAULT 0,
    duration            Float64 DEFAULT 0,
    modified_duration   Float64 DEFAULT 0,
    convexity           Float64 DEFAULT 0,
    
    extra               Map(LowCardinality(String), Float64),
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
ORDER BY (symbol, issuance_date)
SETTINGS index_granularity = 8192;

-- ========================================
-- 10. FACT: Forecast (NEW)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_forecast (
    symbol              LowCardinality(String),
    year                UInt16,
    quarter             UInt8 DEFAULT 0,             -- 0 = yearly
    scenario            LowCardinality(String),      -- BASE / BULL / BEAR
    kpi                 LowCardinality(String),      -- revenue / eps / roe / wacc / beta
    value               Float64 DEFAULT 0,
    confidence_lower    Float64 DEFAULT 0,           -- ADDED: confidence interval
    confidence_upper    Float64 DEFAULT 0,           -- ADDED
    source              LowCardinality(String),      -- AI / Analyst / Broker / User
    model_version       String,
    note                String,
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
ORDER BY (symbol, year, quarter, scenario, kpi)
SETTINGS index_granularity = 8192;

-- ========================================
-- 11. FACT: Budget (NEW)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_budget (
    symbol              LowCardinality(String),
    period_id           UUID,
    budget_revenue      Float64 DEFAULT 0,
    budget_profit       Float64 DEFAULT 0,
    budget_capex        Float64 DEFAULT 0,
    budget_opex         Float64 DEFAULT 0,
    created_by          String,
    created_at          DateTime DEFAULT now()
) ENGINE = MergeTree
ORDER BY (symbol, period_id)
SETTINGS index_granularity = 8192;

-- ========================================
-- 12. SECTOR BENCHMARKS (Aggregating MergeTree)
-- ========================================
CREATE TABLE IF NOT EXISTS fact_sector_benchmark (
    sector              LowCardinality(String),
    year                UInt16,
    quarter             UInt8,
    
    -- Aggregated metrics
    avg_roe             Float32 DEFAULT 0,
    median_roe          Float32 DEFAULT 0,
    avg_pe              Float32 DEFAULT 0,
    median_pe           Float32 DEFAULT 0,
    avg_gross_margin    Float32 DEFAULT 0,
    avg_net_margin      Float32 DEFAULT 0,
    avg_debt_to_equity  Float32 DEFAULT 0,
    
    company_count       UInt32 DEFAULT 0,
    created_at          DateTime DEFAULT now()
) ENGINE = AggregatingMergeTree()
ORDER BY (sector, year, quarter)
SETTINGS index_granularity = 8192;

-- ========================================
-- 13. MASTER ANALYSIS MART
-- ========================================
CREATE TABLE IF NOT EXISTS mart_master_analysis (
    symbol              LowCardinality(String),
    year                UInt16,
    quarter             UInt8,
    report_date         Date,
    
    -- Price
    price               Float32 DEFAULT 0,
    
    -- === VALUATION (10 metrics) ===
    pe_ttm              Float32 DEFAULT 0,
    pb                  Float32 DEFAULT 0,
    ps                  Float32 DEFAULT 0,
    pcf                 Float32 DEFAULT 0,
    peg                 Float32 DEFAULT 0,
    ev_ebitda           Float32 DEFAULT 0,
    ev_sales            Float32 DEFAULT 0,
    market_cap_b        Float32 DEFAULT 0,
    enterprise_value    Float64 DEFAULT 0,
    dividend_yield      Float32 DEFAULT 0,
    
    -- === PROFITABILITY (15 metrics) ===
    roe_ttm             Float32 DEFAULT 0,
    roa_ttm             Float32 DEFAULT 0,
    roic                Float32 DEFAULT 0,
    gross_margin        Float32 DEFAULT 0,
    operating_margin    Float32 DEFAULT 0,
    net_margin          Float32 DEFAULT 0,
    ebitda_margin       Float32 DEFAULT 0,
    asset_turnover      Float32 DEFAULT 0,
    equity_multiplier   Float32 DEFAULT 0,
    dupont_roe          Float32 DEFAULT 0,
    
    -- === GROWTH (10 metrics) ===
    revenue_growth_yoy  Float32 DEFAULT 0,
    profit_growth_yoy   Float32 DEFAULT 0,
    eps_growth_yoy      Float32 DEFAULT 0,
    asset_growth_yoy    Float32 DEFAULT 0,
    equity_growth_yoy   Float32 DEFAULT 0,
    revenue_cagr_3y     Float32 DEFAULT 0,
    profit_cagr_3y      Float32 DEFAULT 0,
    eps_cagr_3y         Float32 DEFAULT 0,
    
    -- === LEVERAGE & LIQUIDITY (12 metrics) ===
    debt_to_equity      Float32 DEFAULT 0,
    debt_to_assets      Float32 DEFAULT 0,
    interest_coverage   Float32 DEFAULT 0,
    current_ratio       Float32 DEFAULT 0,
    quick_ratio         Float32 DEFAULT 0,
    cash_ratio          Float32 DEFAULT 0,
    working_capital     Float64 DEFAULT 0,
    net_debt            Float64 DEFAULT 0,
    net_debt_to_ebitda  Float32 DEFAULT 0,
    
    -- === CASH FLOW QUALITY (8 metrics) ===
    fcf_ttm             Float64 DEFAULT 0,
    fcf_yield           Float32 DEFAULT 0,
    fcf_conversion      Float32 DEFAULT 0,
    cfo_to_revenue      Float32 DEFAULT 0,
    capex_to_revenue    Float32 DEFAULT 0,
    accrual_ratio       Float32 DEFAULT 0,
    
    -- === EFFICIENCY (8 metrics) ===
    receivables_turnover Float32 DEFAULT 0,
    inventory_turnover  Float32 DEFAULT 0,
    payables_turnover   Float32 DEFAULT 0,
    days_receivables    Float32 DEFAULT 0,
    days_inventory      Float32 DEFAULT 0,
    days_payables       Float32 DEFAULT 0,
    cash_conversion_cycle Float32 DEFAULT 0,
    
    -- === QUALITY SCORES (5 metrics) ===
    piotroski_f_score   UInt8 DEFAULT 0,
    altman_z_score      Float32 DEFAULT 0,
    beneish_m_score     Float32 DEFAULT 0,
    sloan_ratio         Float32 DEFAULT 0,
    
    -- === MARKET DATA (from fact_risk_metrics) ===
    beta                Float32 DEFAULT 0,
    volatility_30d      Float32 DEFAULT 0,
    volume_avg_30d      Float64 DEFAULT 0,
    
    -- === FOREIGN DATA ===
    foreign_ownership   Float32 DEFAULT 0,
    foreign_net_buy_ttm Float64 DEFAULT 0,
    room_left           Float32 DEFAULT 0,
    
    -- === SECTOR COMPARISON (5 metrics) ===
    roe_vs_sector       Float32 DEFAULT 0,
    margin_vs_sector    Float32 DEFAULT 0,
    pe_vs_sector        Float32 DEFAULT 0,
    growth_vs_sector    Float32 DEFAULT 0,
    sector_rank         UInt16 DEFAULT 0,
    
    -- Metadata
    created_at          DateTime DEFAULT now(),
    updated_at          DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYY(report_date)
ORDER BY (symbol, report_date)
SETTINGS index_granularity = 8192;

-- Add projection for screener queries
ALTER TABLE mart_master_analysis ADD PROJECTION proj_pe_sorted
(
    SELECT symbol, pe_ttm, roe_ttm, price, market_cap_b, sector
    ORDER BY pe_ttm
);

ALTER TABLE mart_master_analysis ADD PROJECTION proj_roe_sorted
(
    SELECT symbol, roe_ttm, pe_ttm, price, market_cap_b, sector
    ORDER BY roe_ttm DESC
);

-- ========================================
-- MATERIALIZED VIEW: Auto-populate master analysis
-- ========================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_master_analysis TO mart_master_analysis AS
SELECT
    i.symbol,
    p.year,
    p.quarter,
    i.report_date,
    ifNull(m.close, 0) AS price,
    
    -- === VALUATION ===
    ifNull(m.close / nullIf(i.eps * 4, 0), 0) AS pe_ttm,
    ifNull(m.close / nullIf(b.bvps, 0), 0) AS pb,
    ifNull(m.market_cap / nullIf(i.revenue * 4, 0), 0) AS ps,
    ifNull(m.market_cap / nullIf(c.fcf * 4, 0), 0) AS pcf,
    ifNull((m.close / nullIf(i.eps * 4, 0)) / nullIf((i.revenue - prev_i.lag_rev) / nullIf(prev_i.lag_rev, 0) * 100, 0), 0) AS peg,
    ifNull((m.market_cap + b.total_liabilities - b.cash) / nullIf(i.ebitda * 4, 0), 0) AS ev_ebitda,
    ifNull((m.market_cap + b.total_liabilities - b.cash) / nullIf(i.revenue * 4, 0), 0) AS ev_sales,
    ifNull(m.market_cap / 1000000000, 0) AS market_cap_b,
    ifNull(m.market_cap + b.total_liabilities - b.cash, 0) AS enterprise_value,
    ifNull((c.dividends_paid * 4) / nullIf(m.close * m.shares_outstanding, 0) * 100, 0) AS dividend_yield,
    
    -- === PROFITABILITY ===
    ifNull((i.net_income * 4) / nullIf(b.total_equity, 0) * 100, 0) AS roe_ttm,
    ifNull((i.net_income * 4) / nullIf(b.total_assets, 0) * 100, 0) AS roa_ttm,
    ifNull((i.operating_profit * 4) / nullIf(b.total_assets - b.total_current_liab, 0) * 100, 0) AS roic,
    ifNull(i.gross_profit / nullIf(i.revenue, 0) * 100, 0) AS gross_margin,
    ifNull(i.operating_profit / nullIf(i.revenue, 0) * 100, 0) AS operating_margin,
    ifNull(i.net_income / nullIf(i.revenue, 0) * 100, 0) AS net_margin,
    ifNull(i.ebitda / nullIf(i.revenue, 0) * 100, 0) AS ebitda_margin,
    ifNull((i.revenue * 4) / nullIf(b.total_assets, 0), 0) AS asset_turnover,
    ifNull(b.total_assets / nullIf(b.total_equity, 0), 0) AS equity_multiplier,
    ifNull((i.net_income / nullIf(i.revenue, 0)) * 
           ((i.revenue * 4) / nullIf(b.total_assets, 0)) * 
           (b.total_assets / nullIf(b.total_equity, 0)) * 100, 0) AS dupont_roe,
    
    -- === GROWTH (with safe YoY calculation) ===
    ifNull((i.revenue - prev_i.lag_rev) / nullIf(abs(prev_i.lag_rev), 0) * 100, 0) AS revenue_growth_yoy,
    ifNull((i.net_income - prev_i.lag_profit) / nullIf(abs(prev_i.lag_profit), 0) * 100, 0) AS profit_growth_yoy,
    ifNull((i.eps - prev_i.lag_eps) / nullIf(abs(prev_i.lag_eps), 0) * 100, 0) AS eps_growth_yoy,
    ifNull((b.total_assets - prev_b.lag_assets) / nullIf(prev_b.lag_assets, 0) * 100, 0) AS asset_growth_yoy,
    ifNull((b.total_equity - prev_b.lag_equity) / nullIf(prev_b.lag_equity, 0) * 100, 0) AS equity_growth_yoy,
    0 AS revenue_cagr_3y,
    0 AS profit_cagr_3y,
    0 AS eps_cagr_3y,
    
    -- === LEVERAGE & LIQUIDITY ===
    ifNull(b.total_liabilities / nullIf(b.total_equity, 0), 0) AS debt_to_equity,
    ifNull(b.total_liabilities / nullIf(b.total_assets, 0), 0) AS debt_to_assets,
    ifNull(i.ebit / nullIf(i.interest_expense, 0), 0) AS interest_coverage,
    ifNull(b.total_current_assets / nullIf(b.total_current_liab, 0), 0) AS current_ratio,
    ifNull((b.total_current_assets - b.inventory) / nullIf(b.total_current_liab, 0), 0) AS quick_ratio,
    ifNull(b.cash / nullIf(b.total_current_liab, 0), 0) AS cash_ratio,
    ifNull(b.total_current_assets - b.total_current_liab, 0) AS working_capital,
    ifNull((b.short_term_debt + b.long_term_debt) - b.cash, 0) AS net_debt,
    ifNull(((b.short_term_debt + b.long_term_debt) - b.cash) / nullIf(i.ebitda * 4, 0), 0) AS net_debt_to_ebitda,
    
    -- === CASH FLOW QUALITY ===
    ifNull(c.fcf, 0) AS fcf_ttm,
    ifNull((c.fcf * 4) / nullIf(m.market_cap, 0) * 100, 0) AS fcf_yield,
    ifNull((c.fcf * 4) / nullIf(i.net_income * 4, 0), 0) AS fcf_conversion,
    ifNull((c.cfo * 4) / nullIf(i.revenue * 4, 0), 0) AS cfo_to_revenue,
    ifNull(abs(c.capex) / nullIf(i.revenue * 4, 0), 0) AS capex_to_revenue,
    ifNull(((i.net_income * 4) - (c.cfo * 4)) / nullIf(b.total_assets, 0), 0) AS accrual_ratio,
    
    -- === EFFICIENCY ===
    ifNull((i.revenue * 4) / nullIf((b.receivables + prev_b.lag_receiv) / 2, 0), 0) AS receivables_turnover,
    ifNull((i.cogs * 4) / nullIf((b.inventory + prev_b.lag_invent) / 2, 0), 0) AS inventory_turnover,
    ifNull((i.cogs * 4) / nullIf((b.payables + prev_b.lag_payables) / 2, 0), 0) AS payables_turnover,
    ifNull(365 / nullIf((i.revenue * 4) / nullIf((b.receivables + prev_b.lag_receiv) / 2, 0), 0), 0) AS days_receivables,
    ifNull(365 / nullIf((i.cogs * 4) / nullIf((b.inventory + prev_b.lag_invent) / 2, 0), 0), 0) AS days_inventory,
    ifNull(365 / nullIf((i.cogs * 4) / nullIf((b.payables + prev_b.lag_payables) / 2, 0), 0), 0) AS days_payables,
    ifNull((365 / nullIf((i.revenue * 4) / nullIf((b.receivables + prev_b.lag_receiv) / 2, 0), 0)) +
           (365 / nullIf((i.cogs * 4) / nullIf((b.inventory + prev_b.lag_invent) / 2, 0), 0)) -
           (365 / nullIf((i.cogs * 4) / nullIf((b.payables + prev_b.lag_payables) / 2, 0), 0)), 0) AS cash_conversion_cycle,
    
    -- === QUALITY SCORES ===
    toUInt8(
        if(i.net_income > 0, 1, 0) +
        if(c.cfo > 0, 1, 0) +
        if((i.net_income / nullIf(b.total_assets, 0)) > (prev_i.lag_profit / nullIf(prev_b.lag_assets, 0)), 1, 0) +
        if(c.cfo > i.net_income, 1, 0) +
        if(b.total_liabilities / nullIf(b.total_assets, 0) < prev_b.lag_liab / nullIf(prev_b.lag_assets, 0), 1, 0)
    ) AS piotroski_f_score,
    
    ifNull((1.2 * (b.total_current_assets - b.total_current_liab) / nullIf(b.total_assets, 0)) +
           (1.4 * b.retained_earnings / nullIf(b.total_assets, 0)) +
           (3.3 * i.ebit / nullIf(b.total_assets, 0)) +
           (0.6 * m.market_cap / nullIf(b.total_liabilities, 0)) +
           (1.0 * i.revenue / nullIf(b.total_assets, 0)), 0) AS altman_z_score,
    
    0 AS beneish_m_score,
    0 AS sloan_ratio,
    
    -- === MARKET DATA (from risk metrics table via LEFT JOIN) ===
    ifNull(r.beta, 0) AS beta,
    ifNull(r.volatility_30d, 0) AS volatility_30d,
    ifNull(r.volume_avg_30d, 0) AS volume_avg_30d,
    
    -- === FOREIGN DATA ===
    ifNull(m.foreign_buy / nullIf(m.volume, 0) * 100, 0) AS foreign_ownership,
    ifNull(m.foreign_net_buy, 0) AS foreign_net_buy_ttm,
    ifNull(m.room_left, 0) AS room_left,
    
    -- === SECTOR COMPARISON (via LEFT JOIN to sector benchmark) ===
    ifNull((i.net_income * 4) / nullIf(b.total_equity, 0) * 100 - sb.avg_roe, 0) AS roe_vs_sector,
    ifNull(i.net_income / nullIf(i.revenue, 0) * 100 - sb.avg_net_margin, 0) AS margin_vs_sector,
    ifNull(m.close / nullIf(i.eps * 4, 0) - sb.avg_pe, 0) AS pe_vs_sector,
    ifNull((i.revenue - prev_i.lag_rev) / nullIf(abs(prev_i.lag_rev), 0) * 100, 0) AS growth_vs_sector,
    0 AS sector_rank,
    
    now() AS created_at,
    now() AS updated_at

FROM fact_income_statement i
INNER JOIN dim_period p ON i.period_id = p.period_id
LEFT JOIN fact_balance_sheet b ON i.symbol = b.symbol AND i.period_id = b.period_id
LEFT JOIN fact_cash_flow c ON i.symbol = c.symbol AND i.period_id = c.period_id
LEFT JOIN fact_daily_market m ON i.symbol = m.symbol AND m.date = i.report_date

-- Use period mapping for YoY (FIXED: no subquery in JOIN)
LEFT JOIN dim_period_mapping pm ON pm.current_period_id = i.period_id
LEFT JOIN fact_income_statement prev_i ON prev_i.symbol = i.symbol AND prev_i.period_id = pm.prev_period_id
LEFT JOIN fact_balance_sheet prev_b ON prev_b.symbol = i.symbol AND prev_b.period_id = pm.prev_period_id

-- Join risk metrics
LEFT JOIN fact_risk_metrics r ON r.symbol = i.symbol AND r.date = i.report_date

-- Join sector benchmark
LEFT JOIN dim_company dc ON dc.symbol = i.symbol
LEFT JOIN fact_sector_benchmark sb ON sb.sector = dc.sector AND sb.year = p.year AND sb.quarter = p.quarter

WHERE p.period_type IN ('Q', 'YTD')
  AND i.report_date >= '2020-01-01';

-- ========================================
-- MATERIALIZED VIEW: Sector Benchmark
-- ========================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_sector_benchmark TO fact_sector_benchmark AS
SELECT
    c.sector,
    p.year,
    p.quarter,
    
    avg(m.roe_ttm) AS avg_roe,
    quantile(0.5)(m.roe_ttm) AS median_roe,
    avg(m.pe_ttm) AS avg_pe,
    quantile(0.5)(m.pe_ttm) AS median_pe,
    avg(m.gross_margin) AS avg_gross_margin,
    avg(m.net_margin) AS avg_net_margin,
    avg(m.debt_to_equity) AS avg_debt_to_equity,
    
    count() AS company_count,
    now() AS created_at

FROM mart_master_analysis m
INNER JOIN dim_company c ON c.symbol = m.symbol
INNER JOIN dim_period p ON p.year = m.year AND p.quarter = m.quarter
GROUP BY c.sector, p.year, p.quarter;

-- ========================================
-- AUDIT LOG TABLE (Enterprise-grade)
-- ========================================
CREATE TABLE IF NOT EXISTS audit_log (
    log_id              UUID DEFAULT generateUUIDv7(),
    user_id             String,
    action              LowCardinality(String),      -- SELECT / INSERT / UPDATE / DELETE
    table_name          LowCardinality(String),
    query_hash          String,
    query_text          String,
    rows_affected       UInt64,
    execution_time_ms   UInt32,
    ip_address          String,
    timestamp           DateTime DEFAULT now()
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id)
SETTINGS index_granularity = 8192;

-- ========================================
-- SAMPLE DATA INSERTION
-- ========================================

-- Insert sample companies
INSERT INTO dim_company VALUES
('HPG', generateUUIDv7(), 'Tập đoàn Hòa Phát', 'Hoa Phat Group', 
 'Steel', 'Materials', 'HOSE', '2007-11-16', 5000000000, 0.85, 0.45, 1, now(), now()),
('VCB', generateUUIDv7(), 'Ngân hàng TMCP Ngoại thương Việt Nam', 'Vietcombank',
 'Banking', 'Financials', 'HOSE', '2009-07-14', 3709087510, 0.65, 0.00, 1, now(), now()),
('VNM', generateUUIDv7(), 'Công ty Cổ phần Sữa Việt Nam', 'Vinamilk',
 'Food Products', 'Consumer Staples', 'HOSE', '2006-01-19', 1745159100, 0.55, 0.38, 1, now(), now());

-- Insert sample periods
INSERT INTO dim_period VALUES
(generateUUIDv7(), 2024, 4, 'Q', '2024-10-01', '2024-12-31', 1, now()),
(generateUUIDv7(), 2024, 3, 'Q', '2024-07-01', '2024-09-30', 0, now()),
(generateUUIDv7(), 2024, 2, 'Q', '2024-04-01', '2024-06-30', 0, now()),
(generateUUIDv7(), 2024, 1, 'Q', '2024-01-01', '2024-03-31', 0, now()),
(generateUUIDv7(), 2024, 0, 'Y', '2024-01-01', '2024-12-31', 0, now()),
(generateUUIDv7(), 2023, 4, 'Q', '2023-10-01', '2023-12-31', 0, now()),
(generateUUIDv7(), 2023, 3, 'Q', '2023-07-01', '2023-09-30', 0, now()),
(generateUUIDv7(), 2023, 2, 'Q', '2023-04-01', '2023-06-30', 0, now()),
(generateUUIDv7(), 2023, 1, 'Q', '2023-01-01', '2023-03-31', 0, now()),
(generateUUIDv7(), 2023, 0, 'Y', '2023-01-01', '2023-12-31', 0, now());

-- Insert sample macro indicators
INSERT INTO dim_macro_indicator VALUES
('VN_CPI_YOY', 'CPI tăng trưởng YoY', 'CPI YoY', '%', 'VN', 'M', 'GSO', 'Inflation', 1, now()),
('VN_GDP_GROWTH', 'GDP tăng trưởng', 'GDP Growth', '%', 'VN', 'Q', 'GSO', 'GDP', 1, now()),
('SBV_RATE', 'Lãi suất tái cấp vốn', 'Refinancing Rate', '%', 'VN', 'M', 'SBV', 'Monetary', 1, now()),
('VN_CREDIT_GROWTH', 'Tăng trưởng tín dụng', 'Credit Growth', '%', 'VN', 'M', 'SBV', 'Credit', 1, now()),
('VN_EXPORT', 'Kim ngạch xuất khẩu', 'Export Value', 'USD billion', 'VN', 'M', 'GSO', 'Trade', 1, now());

-- ========================================
-- HELPER VIEWS FOR COMMON QUERIES
-- ========================================

-- View: Latest quarter data
CREATE VIEW IF NOT EXISTS view_latest_quarter AS
SELECT m.*, c.company_name_vn, c.sector
FROM mart_master_analysis m
INNER JOIN dim_company c ON c.symbol = m.symbol
INNER JOIN dim_period p ON p.year = m.year AND p.quarter = m.quarter
WHERE p.is_latest_quarter = 1;

-- View: TTM metrics (Trailing 12 months)
CREATE VIEW IF NOT EXISTS view_ttm_metrics AS
SELECT 
    symbol,
    sum(revenue) AS revenue_ttm,
    sum(net_income) AS net_income_ttm,
    sum(cfo) AS cfo_ttm,
    sum(fcf) AS fcf_ttm
FROM fact_income_statement i
INNER JOIN fact_cash_flow c USING (symbol, period_id)
WHERE report_date >= today() - INTERVAL 12 MONTH
GROUP BY symbol;

-- ========================================
-- PERFORMANCE OPTIMIZATION COMMANDS
-- ========================================

-- Materialize projections (run after data load)
-- ALTER TABLE mart_master_analysis MATERIALIZE PROJECTION proj_pe_sorted;
-- ALTER TABLE mart_master_analysis MATERIALIZE PROJECTION proj_roe_sorted;

-- Optimize tables (run periodically)
-- OPTIMIZE TABLE fact_daily_market FINAL;
-- OPTIMIZE TABLE mart_master_analysis FINAL;

-- ========================================
-- SECURITY: Row-Level Security Example
-- ========================================

-- Create user roles
-- CREATE ROLE analyst;
-- CREATE ROLE trader;
-- CREATE ROLE admin;

-- Grant permissions
-- GRANT SELECT ON analytics.* TO analyst;
-- GRANT SELECT, INSERT ON analytics.* TO trader;
-- GRANT ALL ON analytics.* TO admin;

-- Row policy example (restrict sector access)
-- CREATE ROW POLICY restrict_banking ON analytics.mart_master_analysis
-- FOR SELECT USING sector != 'Banking' TO analyst;

-- ========================================
-- MONITORING QUERIES
-- ========================================

-- Check table sizes
-- SELECT 
--     table,
--     formatReadableSize(sum(bytes)) AS size,
--     sum(rows) AS rows
-- FROM system.parts
-- WHERE database = 'analytics'
-- GROUP BY table
-- ORDER BY sum(bytes) DESC;

-- Check query performance
-- SELECT 
--     query_duration_ms,
--     read_rows,
--     query
-- FROM system.query_log
-- WHERE type = 'QueryFinish'
-- ORDER BY query_duration_ms DESC
-- LIMIT 10;

SELECT '✅ ClickHouse Financial OLAP Schema initialized successfully!' AS status,
       'Ready for production data ingestion' AS next_step;