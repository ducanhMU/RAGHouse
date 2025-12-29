import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.ollama import OllamaModel

# Cấu hình Logging
logger = logging.getLogger(__name__)

# =============================================================================
# 1. Dependency Injection
# =============================================================================
@dataclass
class FinancialDeps:
    engine: AsyncEngine

# =============================================================================
# 2. System Prompt (The Brain & Map)
# =============================================================================
# Mô tả Schema rút gọn nhưng đầy đủ ngữ nghĩa để LLM hiểu
DB_SCHEMA_DESCRIPTION = """
You are an expert Financial Data Analyst. You have access to a StarRocks OLAP database containing Vietnam Stock Market data.
Your goal is to answer user questions by generating and executing SQL queries.

### DATABASE SCHEMA OVERVIEW

**1. KEY DIMENSIONS (Always join with these):**
- `dim_company` (Hub): Contains `company_key`, `symbol` (Ticker, e.g., 'HPG'), `sector`, `company_name_vn`.
  - **Rule**: Always filter companies by joining this table on `company_key`.

**2. PRE-CALCULATED ANALYSIS (*** USE THIS FIRST ***):**
- `mart_master_analysis`: The most important table. Contains pre-calculated ratios per quarter.
  - Columns: `pe_ttm`, `pb`, `roe_ttm`, `net_margin`, `dividend_yield`, `piotroski_f_score`, `revenue_growth_yoy`, `net_debt_to_ebitda`.
  - **Strategy**: If user asks for "P/E", "ROE", "Growth", or "Financial Health", query this table first! Do not try to calculate P/E from raw price and earnings.

**3. RAW FINANCIAL FACTS (Use for deep-dive):**
- `fact_income_statement`: `revenue`, `net_income`, `ebitda`, `eps`.
- `fact_balance_sheet`: `total_assets`, `total_equity`, `total_liabilities`, `cash`.
- `fact_cash_flow`: `cfo` (Operating), `cfi` (Investing), `cff` (Financing), `fcf` (Free Cash Flow).

**4. MARKET DATA (Stock Prices):**
- `fact_daily_market`: Daily `close`, `volume`, `market_cap`, `foreign_net_buy`.
- `mv_daily_market_summary`: Aggregated daily stats.

**5. MACRO & SECTOR:**
- `fact_macro_timeseries` & `dim_macro_indicator`: GDP, CPI, Interest Rates.
- `fact_sector_benchmark`: Sector averages for ROE, P/E.

### SQL WRITING RULES:
1.  **Dialect**: Use standard **MySQL** syntax (StarRocks is MySQL compatible).
2.  **Joins**: Always use explicit JOINs. Example: `FROM mart_master_analysis m JOIN dim_company c ON m.company_key = c.company_key`.
3.  **Filtering**: 
    - Always filter by `c.symbol = 'XYZ'`.
    - Handle dates using `year` and `quarter` columns for financial reports, or `date` for market data.
4.  **Limits**: Always add `LIMIT 20` unless the user asks for a specific aggregation, to prevent large data dumps.
5.  **Intent**: If the user question is NOT related to data in these tables (e.g., "How to cook rice?"), DO NOT call any tools. Reply with: "NO_INTENT_DETECTED".
"""

# =============================================================================
# 3. Model Selection Logic (Gemini -> Fallback Ollama)
# =============================================================================
def get_model():
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key:
        logger.info("🤖 Agent Strategy: Using Google Gemini 2.0 Flash")
        # Sử dụng Gemini 2.0 Flash (phiên bản mới nhất, tối ưu cho tool use)
        return GeminiModel(
            'gemini-2.0-flash', 
            api_key=gemini_key
        )
    else:
        logger.warning("🤖 Agent Strategy: Fallback to Local Ollama (Llama 3)")
        # Fallback về Llama 3 chạy local
        return OllamaModel(
            model_name='llama3:8b',
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
        )

# Khởi tạo Agent
financial_agent = Agent(
    model=get_model(),
    deps_type=FinancialDeps,
    system_prompt=DB_SCHEMA_DESCRIPTION,
    retries=2 # Cho phép retry nếu tool lỗi
)

# =============================================================================
# 4. Tools (The Hands)
# =============================================================================
@financial_agent.tool
async def execute_sql(ctx: RunContext[FinancialDeps], query: str) -> str:
    """
    Execute a SQL query against the StarRocks database.
    
    Args:
        query: The MySQL-compatible SQL query string.
        
    Returns:
        A string representation of the result rows (List of Dicts) or an error message.
    """
    logger.info(f"🔍 Agent executing SQL: {query}")
    
    # 1. Basic Safety Check (Cơ bản)
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "ALTER", "TRUNCATE", "GRANT"]
    if any(word in query.upper() for word in forbidden_keywords):
        return "Error: Read-only access. Modification queries are not allowed."

    try:
        # 2. Execution
        async with ctx.deps.engine.connect() as conn:
            # Sử dụng text() để bọc query raw
            result = await conn.execute(text(query))
            
            # 3. Fetch Data
            keys = result.keys()
            rows = result.fetchall()
            
            if not rows:
                return "Query executed successfully. Result: No data found matching the criteria."
            
            # 4. Format Output
            # Chuyển đổi thành List[Dict] để LLM dễ đọc hiểu context
            # Giới hạn kích thước dữ liệu trả về cho LLM (Context Window Safety)
            MAX_ROWS_TO_RETURN = 50
            
            results = []
            for i, row in enumerate(rows):
                if i >= MAX_ROWS_TO_RETURN:
                    break
                # Convert row mapping to dict, handle Decimal/Date serialization if needed by str()
                row_dict = {}
                for key, val in zip(keys, row):
                    row_dict[key] = str(val) if val is not None else None
                results.append(row_dict)
                
            output_str = str(results)
            
            if len(rows) > MAX_ROWS_TO_RETURN:
                output_str += f"\n... (Truncated. Total rows: {len(rows)})"
                
            return output_str

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ SQL Execution Error: {error_msg}")
        # Trả về lỗi chi tiết để LLM có thể tự sửa (Self-correction)
        # Ví dụ: nếu lỗi "Unknown column", LLM sẽ đọc được và thử lại với cột khác.
        return f"SQL Execution Error: {error_msg}. Please check the schema and syntax."