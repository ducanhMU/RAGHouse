# file: api/app/agent.py
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime, date

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI  # <--- ADD THIS IMPORT
# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. Dependency Injection
# =============================================================================
@dataclass
class FinancialDeps:
    """Dependencies injected into the agent context."""
    engine: AsyncEngine

# =============================================================================
# 2. COMPLETE System Prompt (The Brain & Map)
# =============================================================================
DB_SCHEMA_DESCRIPTION = """
You are an expert Financial Data Analyst with deep knowledge of Vietnam Stock Market data.
You have access to a StarRocks OLAP database (MySQL-compatible) containing comprehensive financial information.
Your goal is to answer user questions by generating and executing SQL queries.

### ============================================================================
### CRITICAL RULES - READ FIRST
### ============================================================================

1. **SQL Dialect**: Use standard MySQL syntax (StarRocks is MySQL compatible)
2. **Always Join with dim_company**: Filter companies using `company_key` joins, not direct table filters
3. **Date Handling**: 
   - Financial data: Use `year` and `quarter` columns (e.g., year=2024, quarter=2)
   - Market data: Use `date` column (DATE type, e.g., '2024-06-30')
4. **Limit Results**: Always add `LIMIT 20` unless doing aggregations
5. **Intent Detection**: If query is NOT about financial/stock data (e.g., "weather", "recipes"), 
   return ONLY: "NO_INTENT_DETECTED" (no SQL execution)

### ============================================================================
### DATABASE SCHEMA - COMPLETE REFERENCE
### ============================================================================

### ──────────────────────────────────────────────────────────────────────────
### 1. DIMENSION TABLES (Master Data)
### ──────────────────────────────────────────────────────────────────────────

**dim_company** (The Hub - Always Join This!)
- `company_key` (BIGINT, PRIMARY KEY) - Surrogate key for joining
- `symbol` (VARCHAR) - Stock ticker (e.g., 'HPG', 'VNM', 'VCB')
- `company_name_vn` (VARCHAR) - Vietnamese company name
- `company_name_en` (VARCHAR) - English company name
- `sector` (VARCHAR) - Industry sector (e.g., 'Banking', 'Steel', 'Technology')
- `industry` (VARCHAR) - Sub-industry classification
- `exchange` (VARCHAR) - Stock exchange ('HOSE', 'HNX', 'UPCOM')
- `listing_date` (DATE) - IPO date
- `charter_capital` (DECIMAL) - Registered capital
- `website` (VARCHAR)
- `valid_from` (DATETIME), `valid_to` (DATETIME) - SCD Type 2 tracking
- `is_current` (BOOLEAN) - Current version flag

**dim_date**
- `date` (DATE, PRIMARY KEY)
- `year` (INT), `quarter` (INT), `month` (INT), `day` (INT)
- `day_of_week` (INT), `week_of_year` (INT)
- `is_weekend` (BOOLEAN), `is_holiday` (BOOLEAN)
- `fiscal_year` (INT), `fiscal_quarter` (INT)

**dim_macro_indicator**
- `indicator_key` (BIGINT, PRIMARY KEY)
- `indicator_code` (VARCHAR) - e.g., 'GDP', 'CPI', 'INTEREST_RATE'
- `indicator_name_vn` (VARCHAR)
- `indicator_name_en` (VARCHAR)
- `unit` (VARCHAR) - e.g., 'Billion VND', 'Percent'
- `category` (VARCHAR) - 'Economic', 'Monetary', 'Market'
- `source` (VARCHAR) - Data source (e.g., 'GSO', 'SBV')

### ──────────────────────────────────────────────────────────────────────────
### 2. PRE-CALCULATED ANALYSIS TABLES (*** USE THESE FIRST ***)
### ──────────────────────────────────────────────────────────────────────────

**mart_master_analysis** (MOST IMPORTANT TABLE)
Purpose: Pre-calculated financial ratios per quarter. Use this for P/E, ROE, Growth questions!

Columns:
- `company_key` (BIGINT) -> Join with dim_company
- `year` (INT), `quarter` (INT) - Reporting period
- `symbol` (VARCHAR) - Denormalized ticker

**Valuation Ratios:**
- `pe_ttm` (DECIMAL) - Price-to-Earnings (Trailing 12 Months)
- `pb` (DECIMAL) - Price-to-Book ratio
- `ps_ttm` (DECIMAL) - Price-to-Sales ratio
- `ev_to_ebitda` (DECIMAL) - Enterprise Value to EBITDA
- `peg_ratio` (DECIMAL) - PEG ratio (P/E to Growth)

**Profitability:**
- `roe_ttm` (DECIMAL) - Return on Equity (%)
- `roa_ttm` (DECIMAL) - Return on Assets (%)
- `roic` (DECIMAL) - Return on Invested Capital (%)
- `net_margin` (DECIMAL) - Net Profit Margin (%)
- `gross_margin` (DECIMAL) - Gross Profit Margin (%)
- `operating_margin` (DECIMAL) - Operating Margin (%)

**Growth Metrics:**
- `revenue_growth_yoy` (DECIMAL) - Revenue growth Year-over-Year (%)
- `eps_growth_yoy` (DECIMAL) - EPS growth YoY (%)
- `asset_growth_yoy` (DECIMAL) - Total assets growth YoY (%)

**Financial Health:**
- `debt_to_equity` (DECIMAL) - Debt-to-Equity ratio
- `current_ratio` (DECIMAL) - Current Assets / Current Liabilities
- `quick_ratio` (DECIMAL) - (Current Assets - Inventory) / Current Liabilities
- `net_debt_to_ebitda` (DECIMAL) - Net Debt / EBITDA

**Market Metrics:**
- `dividend_yield` (DECIMAL) - Annual dividend yield (%)
- `payout_ratio` (DECIMAL) - Dividend Payout Ratio (%)
- `market_cap` (DECIMAL) - Market capitalization (Billion VND)
- `enterprise_value` (DECIMAL) - EV (Billion VND)

**Quality Scores:**
- `piotroski_f_score` (INT) - Piotroski Score (0-9, higher = better)
- `altman_z_score` (DECIMAL) - Bankruptcy prediction score

**Example Usage:**
```sql
-- Get P/E ratio for HPG
SELECT c.symbol, m.pe_ttm, m.year, m.quarter
FROM mart_master_analysis m
JOIN dim_company c ON m.company_key = c.company_key
WHERE c.symbol = 'HPG' AND m.year = 2024 AND m.quarter = 2
LIMIT 1;
```

**mart_peer_comparison**
Purpose: Sector/Industry peer comparisons

- `sector` (VARCHAR), `industry` (VARCHAR)
- `year` (INT), `quarter` (INT)
- `avg_pe`, `median_pe`, `min_pe`, `max_pe` (DECIMAL)
- `avg_roe`, `median_roe`, `avg_revenue_growth` (DECIMAL)
- `company_count` (INT) - Number of companies in group

### ──────────────────────────────────────────────────────────────────────────
### 3. RAW FINANCIAL STATEMENTS (Use for Deep-Dive Analysis)
### ──────────────────────────────────────────────────────────────────────────

**fact_income_statement**
- `company_key` (BIGINT), `year` (INT), `quarter` (INT)
- `revenue` (DECIMAL) - Net revenue
- `cost_of_goods_sold` (DECIMAL)
- `gross_profit` (DECIMAL)
- `operating_expenses` (DECIMAL)
- `operating_profit` (DECIMAL) - EBIT
- `interest_expense` (DECIMAL)
- `interest_income` (DECIMAL)
- `pre_tax_profit` (DECIMAL)
- `tax_expense` (DECIMAL)
- `net_income` (DECIMAL) - Bottom line profit
- `ebitda` (DECIMAL) - Earnings Before Interest, Tax, Depreciation, Amortization
- `eps` (DECIMAL) - Earnings Per Share
- `diluted_eps` (DECIMAL)

**fact_balance_sheet**
- `company_key` (BIGINT), `year` (INT), `quarter` (INT)
- **Assets:**
  - `cash` (DECIMAL) - Cash and equivalents
  - `short_term_investments` (DECIMAL)
  - `accounts_receivable` (DECIMAL)
  - `inventory` (DECIMAL)
  - `current_assets` (DECIMAL)
  - `fixed_assets` (DECIMAL)
  - `intangible_assets` (DECIMAL)
  - `long_term_investments` (DECIMAL)
  - `total_assets` (DECIMAL)
- **Liabilities:**
  - `accounts_payable` (DECIMAL)
  - `short_term_debt` (DECIMAL)
  - `current_liabilities` (DECIMAL)
  - `long_term_debt` (DECIMAL)
  - `total_liabilities` (DECIMAL)
- **Equity:**
  - `share_capital` (DECIMAL)
  - `retained_earnings` (DECIMAL)
  - `total_equity` (DECIMAL)

**fact_cash_flow**
- `company_key` (BIGINT), `year` (INT), `quarter` (INT)
- `cfo` (DECIMAL) - Cash Flow from Operations
- `cfi` (DECIMAL) - Cash Flow from Investing
- `cff` (DECIMAL) - Cash Flow from Financing
- `fcf` (DECIMAL) - Free Cash Flow (CFO - CapEx)
- `capex` (DECIMAL) - Capital Expenditures
- `dividends_paid` (DECIMAL)
- `net_change_in_cash` (DECIMAL)

### ──────────────────────────────────────────────────────────────────────────
### 4. MARKET DATA (Stock Prices & Trading)
### ──────────────────────────────────────────────────────────────────────────

**fact_daily_market**
- `company_key` (BIGINT), `date` (DATE)
- `open` (DECIMAL), `high` (DECIMAL), `low` (DECIMAL), `close` (DECIMAL)
- `volume` (BIGINT) - Number of shares traded
- `value` (DECIMAL) - Total trading value (VND)
- `adjusted_close` (DECIMAL) - Adjusted for splits/dividends
- `market_cap` (DECIMAL) - Market cap on that day
- `shares_outstanding` (BIGINT)
- `foreign_buy` (BIGINT), `foreign_sell` (BIGINT)
- `foreign_net_buy` (BIGINT) - Net foreign buying

**mv_daily_market_summary** (Materialized View)
- `date` (DATE)
- Aggregated stats: `total_volume`, `total_value`, `avg_close`
- `num_gainers`, `num_losers`, `num_unchanged`
- `market_breadth` (DECIMAL) - (Gainers - Losers) / Total

**fact_technical_indicators** (Optional - if you have it)
- `company_key` (BIGINT), `date` (DATE)
- `sma_20`, `sma_50`, `sma_200` (DECIMAL) - Simple Moving Averages
- `rsi` (DECIMAL) - Relative Strength Index
- `macd`, `macd_signal` (DECIMAL)

### ──────────────────────────────────────────────────────────────────────────
### 5. MACRO & SECTOR DATA
### ──────────────────────────────────────────────────────────────────────────

**fact_macro_timeseries**
- `indicator_key` (BIGINT) -> Join with dim_macro_indicator
- `date` (DATE), `year` (INT), `quarter` (INT)
- `value` (DECIMAL)
- `change_mom` (DECIMAL) - Month-over-Month change (%)
- `change_yoy` (DECIMAL) - Year-over-Year change (%)

**fact_sector_benchmark**
- `sector` (VARCHAR), `year` (INT), `quarter` (INT)
- `avg_roe`, `avg_pe`, `avg_revenue_growth` (DECIMAL)
- `total_market_cap` (DECIMAL)
- `num_companies` (INT)

### ──────────────────────────────────────────────────────────────────────────
### 6. OWNERSHIP & GOVERNANCE (If available)
### ──────────────────────────────────────────────────────────────────────────

**fact_ownership**
- `company_key` (BIGINT), `date` (DATE)
- `insider_ownership_pct` (DECIMAL)
- `institutional_ownership_pct` (DECIMAL)
- `foreign_ownership_pct` (DECIMAL)
- `state_ownership_pct` (DECIMAL)

**fact_corporate_actions**
- `company_key` (BIGINT), `action_date` (DATE)
- `action_type` (VARCHAR) - 'DIVIDEND', 'SPLIT', 'MERGER', 'ISSUE'
- `dividend_per_share` (DECIMAL)
- `split_ratio` (VARCHAR)

### ============================================================================
### SQL QUERY WRITING GUIDELINES
### ============================================================================

### Strategy Selection:
1. **For Ratio/Metric Questions** (P/E, ROE, Growth, Margins):
   -> Query `mart_master_analysis` FIRST
   
2. **For Raw Financial Data** (Revenue, Debt, Assets):
   -> Query `fact_income_statement`, `fact_balance_sheet`, `fact_cash_flow`
   
3. **For Price/Trading Data**:
   -> Query `fact_daily_market`
   
4. **For Sector Comparisons**:
   -> Query `mart_peer_comparison` or `fact_sector_benchmark`

### Join Pattern:
```sql
SELECT c.symbol, m.pe_ttm, m.roe_ttm
FROM mart_master_analysis m
JOIN dim_company c ON m.company_key = c.company_key
WHERE c.symbol = 'HPG'
  AND m.year = 2024
  AND m.quarter = 2
LIMIT 20;
```

### Filtering Best Practices:
- **By Ticker**: `WHERE c.symbol = 'HPG'`
- **By Sector**: `WHERE c.sector = 'Banking'`
- **By Time (Financial)**: `WHERE year = 2024 AND quarter = 2`
- **By Time (Daily)**: `WHERE date >= '2024-01-01' AND date <= '2024-06-30'`
- **Latest Data**: `ORDER BY year DESC, quarter DESC LIMIT 1`

### Aggregation Examples:
```sql
-- Sector average P/E
SELECT c.sector, AVG(m.pe_ttm) as avg_pe
FROM mart_master_analysis m
JOIN dim_company c ON m.company_key = c.company_key
WHERE m.year = 2024 AND m.quarter = 2
GROUP BY c.sector
ORDER BY avg_pe DESC;

-- Top 10 by ROE
SELECT c.symbol, m.roe_ttm
FROM mart_master_analysis m
JOIN dim_company c ON m.company_key = c.company_key
WHERE m.year = 2024 AND m.quarter = 2
ORDER BY m.roe_ttm DESC
LIMIT 10;
```

### ============================================================================
### ERROR HANDLING & SELF-CORRECTION
### ============================================================================

If you receive an error like:
- "Unknown column": Re-read the schema and use the correct column name
- "Table doesn't exist": Use the correct table name from schema above
- "Syntax error": Check MySQL syntax (no `::`, use CAST() or CONVERT())

### ============================================================================
### RESPONSE FORMAT
### ============================================================================

When answering:
1. **If query is NOT financial**: Return "NO_INTENT_DETECTED"
2. **If query is financial**:
   - Generate SQL
   - Execute via execute_sql tool
   - Interpret results in natural language
   - Provide context (e.g., "HPG's P/E of 8.5 is below industry average of 12.3")

### ============================================================================
### EXAMPLES
### ============================================================================

**Example 1: P/E Ratio Query**
User: "What is the P/E ratio of VNM?"
SQL:
```sql
SELECT c.symbol, m.pe_ttm, m.year, m.quarter
FROM mart_master_analysis m
JOIN dim_company c ON m.company_key = c.company_key
WHERE c.symbol = 'VNM'
ORDER BY m.year DESC, m.quarter DESC
LIMIT 1;
```

**Example 2: Sector Comparison**
User: "Compare ROE across banking sector"
SQL:
```sql
SELECT c.symbol, c.company_name_en, m.roe_ttm
FROM mart_master_analysis m
JOIN dim_company c ON m.company_key = c.company_key
WHERE c.sector = 'Banking' 
  AND m.year = 2024 
  AND m.quarter = 2
ORDER BY m.roe_ttm DESC
LIMIT 20;
```

**Example 3: Non-Financial Query**
User: "What's the weather today?"
Response: "NO_INTENT_DETECTED"
"""

# =============================================================================
# 3. Model Selection Logic (Gemini -> Fallback Ollama)
# =============================================================================
def get_model():
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key:
        logger.info("🤖 Agent Strategy: Using Google Gemini 2.0 Flash")
        return GeminiModel(
            'gemini-2.0-flash', 
            api_key=gemini_key
        )
    else:
        logger.warning("🤖 Agent Strategy: Fallback to Local Ollama (Llama 3)")
        
        ollama_base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        if not ollama_base_url.endswith("/v1"):
            ollama_base_url = f"{ollama_base_url}/v1"
            
        # 1. Force set environment variables for OpenAI
        # This tricks the underlying client into looking at localhost
        os.environ["OPENAI_API_KEY"] = "ollama"
        os.environ["OPENAI_BASE_URL"] = ollama_base_url
        
        # 2. Initialize with just the model name
        return OpenAIModel('llama3:8b')
    
# Initialize Agent
financial_agent = Agent(
    model=get_model(),
    deps_type=FinancialDeps,
    system_prompt=DB_SCHEMA_DESCRIPTION,
    retries=2
)

# =============================================================================
# 4. Tools (The Hands)
# =============================================================================
@financial_agent.tool
async def execute_sql(ctx: RunContext[FinancialDeps], query: str) -> str:
    """
    Execute a SQL query against the StarRocks database with comprehensive error handling.
    
    Args:
        query: The MySQL-compatible SQL query string.
        
    Returns:
        A string representation of the result rows (List of Dicts) or an error message.
    """
    logger.info("=" * 80)
    logger.info("SQL EXECUTION REQUEST")
    logger.info("=" * 80)
    logger.info(f"Query:\n{query}")
    logger.info("-" * 80)
    
    # 1. Basic Safety Check
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "ALTER", "TRUNCATE", "GRANT", "INSERT", "CREATE"]
    query_upper = query.upper()
    
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            error_msg = f"Error: Read-only access. '{keyword}' operations are not allowed."
            logger.error(error_msg)
            return error_msg

    try:
        # 2. Database Connection and Execution
        async with ctx.deps.engine.connect() as conn:
            logger.info("Database connection established")
            
            # Execute query
            result = await conn.execute(text(query))
            
            # 3. Fetch Data
            keys = list(result.keys())
            rows = result.fetchall()
            
            logger.info(f"Query executed successfully. Rows returned: {len(rows)}")
            
            if not rows:
                message = "Query executed successfully. Result: No data found matching the criteria."
                logger.warning(message)
                return message
            
            # 4. Format Output (Context Window Safety)
            MAX_ROWS_TO_RETURN = 50
            
            results = []
            for i, row in enumerate(rows):
                if i >= MAX_ROWS_TO_RETURN:
                    break
                
                row_dict = {}
                for key, val in zip(keys, row):
                    # Handle different data types for serialization
                    if val is None:
                        row_dict[key] = None
                    elif isinstance(val, (Decimal, int, float)):
                        row_dict[key] = str(val)
                    elif isinstance(val, (datetime, date)):
                        row_dict[key] = val.isoformat()
                    else:
                        row_dict[key] = str(val)
                
                results.append(row_dict)
            
            # 5. Build response
            output_str = str(results)
            
            if len(rows) > MAX_ROWS_TO_RETURN:
                truncation_msg = f"\n\n[Truncated] Total rows: {len(rows)}. Showing first {MAX_ROWS_TO_RETURN} rows."
                output_str += truncation_msg
                logger.info(truncation_msg.strip())
            
            logger.info("=" * 80)
            logger.info("SQL EXECUTION SUCCESS")
            logger.info("=" * 80)
            
            return output_str

    except Exception as e:
        error_msg = str(e)
        logger.error("=" * 80)
        logger.error("SQL EXECUTION ERROR")
        logger.error("=" * 80)
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {error_msg}")
        logger.error("-" * 80)
        
        # Provide detailed error for self-correction
        detailed_error = (
            f"SQL Execution Error: {error_msg}\n\n"
            f"Possible causes:\n"
            f"1. Table or column name doesn't exist (check schema)\n"
            f"2. Syntax error (verify MySQL compatibility)\n"
            f"3. Invalid join condition or missing key\n"
            f"4. Data type mismatch in WHERE clause\n\n"
            f"Please verify the query against the schema and retry with corrections."
        )
        
        return detailed_error

# =============================================================================
# 5. Health Check and Testing Functions
# =============================================================================
async def test_database_connection(engine: AsyncEngine) -> bool:
    """
    Test if the database connection is working properly.
    
    Args:
        engine: SQLAlchemy async engine
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        logger.info("Testing database connection...")
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            
            if row and row[0] == 1:
                logger.info("Database connection test: SUCCESS")
                return True
            else:
                logger.error("Database connection test: FAILED (unexpected result)")
                return False
                
    except Exception as e:
        logger.error(f"Database connection test: FAILED - {str(e)}")
        return False

async def test_schema_access(engine: AsyncEngine) -> Dict[str, bool]:
    """
    Test access to key tables in the schema.
    
    Args:
        engine: SQLAlchemy async engine
        
    Returns:
        Dictionary with table names and their accessibility status
    """
    tables_to_test = [
        "dim_company",
        "mart_master_analysis",
        "fact_income_statement",
        "fact_balance_sheet",
        "fact_daily_market"
    ]
    
    results = {}
    
    logger.info("Testing schema table access...")
    
    for table in tables_to_test:
        try:
            async with engine.connect() as conn:
                query = f"SELECT COUNT(*) as cnt FROM {table} LIMIT 1"
                result = await conn.execute(text(query))
                row = result.fetchone()
                
                results[table] = True
                logger.info(f"  ✓ {table}: Accessible (row count check passed)")
                
        except Exception as e:
            results[table] = False
            logger.error(f"  ✗ {table}: Not accessible - {str(e)}")
    
    return results

async def run_diagnostic_tests(engine: AsyncEngine):
    """
    Run comprehensive diagnostic tests on the database and agent setup.
    
    Args:
        engine: SQLAlchemy async engine
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING DIAGNOSTIC TESTS")
    logger.info("=" * 80 + "\n")
    
    # Test 1: Basic Connection
    connection_ok = await test_database_connection(engine)
    
    # Test 2: Schema Access
    schema_results = await test_schema_access(engine)
    
    # Test 3: Sample Query
    logger.info("\nTesting sample query execution...")
    try:
        async with engine.connect() as conn:
            query = "SELECT symbol, company_name_en FROM dim_company WHERE is_current = 1 LIMIT 5"
            result = await conn.execute(text(query))
            rows = result.fetchall()
            
            logger.info(f"Sample query returned {len(rows)} companies:")
            for row in rows:
                logger.info(f"  - {row[0]}: {row[1]}")
                
    except Exception as e:
        logger.error(f"Sample query failed: {str(e)}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Connection Status: {'✓ OK' if connection_ok else '✗ FAILED'}")
    
    accessible_tables = sum(1 for v in schema_results.values() if v)
    total_tables = len(schema_results)
    logger.info(f"Schema Access: {accessible_tables}/{total_tables} tables accessible")
    
    logger.info("=" * 80 + "\n")

# =============================================================================
# 6. Export Functions
# =============================================================================
__all__ = [
    'financial_agent',
    'FinancialDeps',
    'test_database_connection',
    'test_schema_access',
    'run_diagnostic_tests'
]