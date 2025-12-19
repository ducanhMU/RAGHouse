# 📌 FINAL REFINED FINANCIAL ANALYSIS RAG USE CASE SPEC

Tài liệu này mô tả chi tiết thiết kế các use case cho hệ thống RAG (Retrieval-Augmented Generation) hỗ trợ phân tích tài chính doanh nghiệp. Hệ thống kết hợp dữ liệu OLAP từ ClickHouse (các bảng như `dim_company, dim_period, fact_income_statement, fact_balance_sheet, fact_cash_flow, fact_daily_market, dim_macro_indicator, fact_macro_timeseries, mart_master_analysis`, và các bảng bổ sung như `bond_data, forecast_table, budget_table`), thuật toán tính toán KPI (pre-calculated trong `mart_master_analysis` qua materialized view `mv_master_analysis`, hoặc on-fly qua `code_execution`), và generation narrative dựa trên RAG. Các use case được phân loại theo persona người dùng (`Analyst, Trader/Investor, CEO/CFO/Manager, General User`), với chi tiết input/output, logic tính toán dựa trên tables, edge cases, và RAG prompt. Hệ thống tránh tư vấn pháp lý, tập trung dữ liệu khách quan.

## 📊 Glossary (Chỉ số KPI)
| KPI | Công thức | Mô tả | Database Link & Logic |
|-----|-----------|-------|-----------------------|
| PE TTM | price / (eps*4) | Định giá dựa trên lợi nhuận trailing twelve months. | mart_master_analysis.pe_ttm; Logic: Join fact_daily_market.close và fact_income_statement.eps, sum 4 quarters cho TTM nếu cần. |
| PB | price / bvps | Định giá dựa trên giá trị sổ sách. | mart_master_analysis.pb; Logic: fact_daily_market.close / fact_balance_sheet.bvps. |
| ROE | NI*4 / equity*100 | Hiệu quả sinh lời trên vốn chủ sở hữu. | mart_master_analysis.roe_ttm; Logic: fact_income_statement.net_income *4 / fact_balance_sheet.total_equity. |
| FCF Yield | FCF*4 / market_cap*100 | Lợi suất dòng tiền tự do. | mart_master_analysis.fcf_yield; Logic: fact_cash_flow.fcf *4 / fact_daily_market.market_cap. |
| Volatility | STDDEV(close) | Độ biến động giá cổ phiếu. | On-fly từ fact_daily_market.close; Logic: STDDEV over window (e.g., 30 days). |
| ROA TTM | NI*4 / total_assets*100 | Hiệu quả sinh lời trên tài sản. | mart_master_analysis.roa_ttm; Logic: fact_income_statement.net_income *4 / fact_balance_sheet.total_assets. |
| ROIC | NOPAT / invested_capital | Hiệu quả sinh lời trên vốn đầu tư. | mart_master_analysis.roic; Logic: (fact_income_statement.operating_profit * (1 - tax_rate)) / (fact_balance_sheet.total_assets - total_current_liab). |
| EV/EBITDA | (market_cap + debt - cash) / EBITDA | Định giá doanh nghiệp. | mart_master_analysis.ev_ebitda; Logic: (fact_daily_market.market_cap + fact_balance_sheet.short_term_debt + long_term_debt - cash) / fact_income_statement.ebitda. |
| Dividend Yield | dividends / price * 100 | Lợi suất cổ tức. | mart_master_analysis.dividend_yield; Logic: fact_cash_flow.dividends_paid / fact_daily_market.close *100. |
| Growth YoY | (current - previous) / previous * 100 | Tăng trưởng năm trên năm. | mart_master_analysis.revenue_growth_yoy etc.; Logic: LAG(revenue) over ORDER BY year, quarter từ mart_master_analysis. |
| CFO/Revenue | Cash from Operations / Revenue | Chất lượng dòng tiền hoạt động. | mart_master_analysis.cfo_to_revenue; Logic: fact_cash_flow.cfo / fact_income_statement.revenue. |
| Accrual Ratio | (NI - CFO) / total_assets | Chất lượng lợi nhuận. | mart_master_analysis.accrual_ratio; Logic: (fact_income_statement.net_income - fact_cash_flow.cfo) / fact_balance_sheet.total_assets. |
| Current Ratio | Current Assets / Current Liabilities | Khả năng thanh toán ngắn hạn. | mart_master_analysis.current_ratio; Logic: fact_balance_sheet.total_current_assets / total_current_liab. |
| D/E | Debt / Equity | Rủi ro nợ. | mart_master_analysis.debt_to_equity; Logic: (fact_balance_sheet.short_term_debt + long_term_debt) / total_equity. |
| Cash Conversion Cycle | DIO + DSO - DPO | Hiệu quả vốn lưu động. | mart_master_analysis.cash_conversion_cycle; Logic: mart_master_analysis.days_inventory + days_sales - days_payables. |
| Altman Z-Score | 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MV/BV) + 1.0*(Sales/TA) | Dự báo rủi ro phá sản. | mart_master_analysis.altman_z_score; Logic: Tính từ fact_balance_sheet (WC=total_current_assets - liab, RE=retained_earnings, TA=total_assets) + fact_income_statement (EBIT, Sales=revenue) + fact_daily_market (MV=market_cap, BV=total_equity). |
| RSI | 100 - (100 / (1 + RS)), RS = Avg Gain / Avg Loss | Chỉ báo kỹ thuật. | On-fly code_execution từ fact_daily_market.close (over 14 days). |
| MACD | EMA12 - EMA26 | Chỉ báo động lượng. | On-fly code_execution (numpy.ema) từ fact_daily_market.close. |
| YTM | Solve NPV=0 for r | Lợi suất đến hạn trái phiếu. | On-fly code_execution (scipy.optimize) từ bond_data (cash_flows Array, coupon_rate, maturity_date). |
| IRR | Rate where NPV=0 | Tỷ suất hoàn vốn nội bộ. | On-fly code_execution (numpy.irr) từ bond_data cash_flows. |
| NPV | Sum(CF_t / (1+r)^t) - Initial | Giá trị hiện tại ròng. | On-fly code_execution (numpy.npv) từ bond_data. |
| Kd | Interest Expense / Total Debt * (1 - Tax Rate) | Chi phí nợ. | On-fly từ fact_income_statement.interest_expense / (fact_balance_sheet.short_term_debt + long_term_debt) * (1 - tax từ extra_items hoặc default 0.2). |
| Ke | Rf + Beta * (Rm - Rf) | Chi phí vốn chủ. | On-fly từ fact_macro_timeseries (Rf=SBV_RATE, Rm=market return), mart_master_analysis.beta. |
| WACC | (E/V * Ke) + (D/V * Kd * (1 - Tax)) | Chi phí vốn trung bình. | On-fly từ Kd, Ke, fact_balance_sheet (E=total_equity, D=debt, V=E+D). |
| Interest Coverage | EBIT / Interest Expense | Khả năng trả lãi. | mart_master_analysis.interest_coverage; Logic: fact_income_statement.ebit / interest_expense. |
| Short-Term Debt Ratio | Short-Term Debt / Total Debt | Tỷ trọng nợ ngắn hạn. | On-fly fact_balance_sheet.short_term_debt / (short_term_debt + long_term_debt). |
| Long-Term Debt Ratio | Long-Term Debt / Total Debt | Tỷ trọng nợ dài hạn. | Tương tự trên. |
| Beneish M-Score | Complex (DSRI + GMI + AQI + SGI + DEPI + SGAI + TATA + LVGI) | Phát hiện gian lận. | On-fly code_execution từ mart_master_analysis (tăng trưởng assets, margins, etc.). |

**Cash Flow Phân loại:** CFO (fact_cash_flow.cfo), CFI (cfi), CFF (cff); Logic: Phân tích sign (dương/âm) và tỷ trọng.

## ===========================
## A. Người phân tích / Chuyên gia tài chính (Analyst)
## ===========================
### **Use Case A1 — Tóm tắt tình hình tài chính doanh nghiệp**
**Mục tiêu nghiệp vụ:** Cung cấp dashboard summary toàn diện về IS/BS/CF/market, với narrative. Edge: Kỳ không tồn tại → fallback latest; multi-ticker → aggregate.

**Input:** Ticker (dim_company.symbol, e.g., 'VCB'), Kỳ (dim_period: year=2024, quarter=4, period_type='Q'/'YTD'/'TTM'/'Y').

**Output:** Bảng Markdown/JSON: revenue (fact_income_statement), net_income, eps, roe_ttm (mart_master_analysis), roa_ttm, cfo_to_revenue, gross_margin, market_cap_b (mart_master_analysis); CFO/CFI/CFF (fact_cash_flow); short/long debt ratio; narrative (e.g., "Revenue tăng, debt cao rủi ro").

**OLAP/Data Source:** mart_master_analysis (core KPIs), join dim_period cho filter, fact_daily_market cho price update, fact_cash_flow cho phân loại, fact_balance_sheet cho debt.

**Logic/Tính Toán:** Pre-calc từ mart; TTM: SUM(revenue) over last 4 quarters (WINDOW function in SQL); Debt ratios: short_term_debt / total_debt. Edge: Null → 0 hoặc note; code_execution nếu sum custom.

**Example SQL:** 
```sql
SELECT revenue, net_income, eps, roe_ttm, roa_ttm, cfo_to_revenue, gross_margin, market_cap_b, 
c.cfo, c.cfi, c.cff, b.short_term_debt / (b.short_term_debt + b.long_term_debt) AS short_ratio
FROM mart_master_analysis m 
JOIN fact_cash_flow c ON m.symbol=c.symbol AND m.report_date=c.report_date
JOIN fact_balance_sheet b ON m.symbol=b.symbol AND m.report_date=b.report_date
WHERE m.symbol='VCB' AND m.year=2024 AND m.quarter=4;
-- TTM: SUM over subtractQuarters(now(),4).
```
**RAG Prompt:** “Tóm tắt tình hình tài chính VCB Q4-2024, bao gồm KPIs, cash flow phân loại, cấu trúc nợ.”

### **Use Case A2 — So sánh chỉ số tài chính giữa các kỳ**
**Mục tiêu nghiệp vụ:** Trend YoY/QoQ, detect anomalies (>20% change). Edge: Missing periods → interpolate/note; custom type (YoY/QoQ).

**Input:** Ticker, List periods (array dim_period: e.g., [(2023,3),(2023,4),(2024,1)]), Comparison_type ('YoY'/'QoQ').

**Output:** Bảng: period_label (year-Qquarter), revenue, growth_yoy (%), net_margin; chart data JSON; alerts (spike/drop); narrative.

**OLAP/Data Source:** mart_master_analysis (growth pre-calc), join dim_period.

**Logic/Tính Toán:** Growth = (cur - LAG(value)) / LAG(value) *100; Anomaly: WHERE abs(growth)>20. Edge: Previous null → 0%.

**Example SQL:** 
```sql
SELECT year || '-Q' || quarter AS label, revenue, revenue_growth_yoy, net_margin
FROM mart_master_analysis WHERE symbol='HPG' AND (year,quarter) IN ((2023,3),(2023,4),(2024,1),(2024,2)) ORDER BY year,quarter;
-- QoQ: LAG over sequential quarters.
```
**RAG Prompt:** “So sánh doanh thu, biên lợi nhuận HPG qua 4 quý gần nhất, detect bất thường.”

### **Use Case A3 — Phân tích hiệu quả hoạt động**
**Mục tiêu nghiệp vụ:** ROE/ROA/ROIC với drivers (DuPont). Edge: Negative → inefficiency alert.

**Input:** Ticker, Period (dim_period: year=2024, quarter=0 for yearly).

**Output:** roe_ttm, roa_ttm, roic; breakdown (net_margin * asset_turnover * equity_multiplier); narrative drivers.

**OLAP/Data Source:** mart_master_analysis (dupont_roe pre-calc).

**Logic/Tính Toán:** ROE = net_margin * asset_turnover * equity_multiplier từ mart.

**Example SQL:** 
```sql
SELECT roe_ttm, roa_ttm, roic, dupont_roe, net_margin, asset_turnover, equity_multiplier
FROM mart_master_analysis WHERE symbol='VCB' AND year=2024 AND quarter=0;
```
**RAG Prompt:** “Phân tích ROE, ROA, ROIC VCB 2024 với DuPont.”

### **Use Case A4 — Định giá doanh nghiệp (Valuation)**
**Mục tiêu nghiệp vụ:** Multiples vs benchmark, WACC cho DCF. Edge: No peers → market avg.

**Input:** Ticker, Sector (dim_company.sector), Period.

**Output:** pe_ttm, pb, ev_ebitda, wacc; deviation vs sector median (%); narrative (undervalued if < median).

**OLAP/Data Source:** mart_master_analysis, aggregate GROUP BY sector từ dim_company.

**Logic/Tính Toán:** Deviation = (company - median) / median *100; WACC on-fly (Kd from interest_expense / debt, Ke CAPM từ beta + macro Rf).

**Example SQL:** 
```sql
WITH sector_med AS (SELECT median(pe_ttm) AS med_pe FROM mart_master_analysis JOIN dim_company ON symbol=symbol WHERE sector='Steel' AND year=2024)
SELECT m.pe_ttm, m.pb, m.ev_ebitda, (b.total_equity / (total_equity + debt) * ke) + (debt / v * kd * (1-0.2)) AS wacc, (m.pe_ttm - s.med_pe)/s.med_pe*100 AS dev
FROM mart_master_analysis m JOIN fact_balance_sheet b ON ... CROSS JOIN sector_med s WHERE m.symbol='HPG' AND m.year=2024;
-- Ke: fact_macro_timeseries.value (Rf) + m.beta * (Rm - Rf).
```
**RAG Prompt:** “Định giá HPG so trung vị ngành thép, dùng WACC từ Kd/Ke.”

### **Use Case A5 — Benchmark so với ngành/sector**
**Mục tiêu nghiệp vụ:** KPI vs sector avg/median, rank. Edge: Multi-KPI custom.

**Input:** Ticker, Sector, Period.

**Output:** Bảng: company_value, sector_avg, deviation (%); rank (1/N); narrative strengths/weaknesses.

**OLAP/Data Source:** mart_master_analysis GROUP BY dim_company.sector.

**Logic/Tính Toán:** Avg/median per sector; Rank: ROW_NUMBER() over ORDER BY pe_ttm ASC.

**Example SQL:** 
```sql
WITH sector_stats AS (SELECT avg(pe_ttm) AS avg_pe, avg(roe_ttm) AS avg_roe FROM mart_master_analysis JOIN dim_company ON ... WHERE sector='Banking' AND year=2024)
SELECT m.pe_ttm, s.avg_pe, (m.pe_ttm - s.avg_pe)/s.avg_pe*100 AS dev FROM mart_master_analysis m CROSS JOIN sector_stats s WHERE m.symbol='VCB';
-- Rank: SELECT symbol, pe_ttm, ROW_NUMBER() OVER (PARTITION BY sector ORDER BY pe_ttm) AS rank FROM ...
```
**RAG Prompt:** “Benchmark ROE, PE VCB so ngân hàng cùng ngành.”

### **Use Case A6 — Phân tích dòng tiền & chất lượng lợi nhuận**
**Mục tiêu nghiệp vụ:** Quality metrics, CFO/CFI/CFF breakdown, lãi suất nợ. Edge: Negative flow → alert.

**Input:** Ticker, Period.

**Output:** cfo_to_revenue, fcf_yield, accrual_ratio; CFO/CFI/CFF values & %; interest_coverage; narrative.

**OLAP/Data Source:** mart_master_analysis + fact_cash_flow.

**Logic/Tính Toán:** % = cfo / (cfo + cfi + cff) *100; Kd on-fly.

**Example SQL:** 
```sql
SELECT cfo_to_revenue, fcf_yield, accrual_ratio, c.cfo, c.cfi, c.cff, interest_coverage
FROM mart_master_analysis m JOIN fact_cash_flow c ON m.symbol=c.symbol AND m.report_date=c.report_date WHERE symbol='VCB' AND year=2024;
```
**RAG Prompt:** “Phân tích dòng tiền CFO/CFI/CFF, chất lượng lợi nhuận VCB 2024, bao lãi suất nợ.”

### **Use Case A7 — Phân tích hoạt động theo phương pháp ngang (Horizontal Analysis)**
**Mục tiêu nghiệp vụ:** % change over periods, anomalies. Edge: Base missing → earliest.

**Input:** Ticker, Multi-periods (array dim_period).

**Output:** Bảng: period, revenue, %change (YoY), expenses, %change; alerts (>20%); narrative.

**OLAP/Data Source:** mart_master_analysis with LAG.

**Logic/Tính Toán:** %change = (cur - lag) / lag *100; Anomaly filter abs>20.

**Example SQL:** 
```sql
SELECT year, quarter, revenue, (revenue - LAG(revenue) OVER (PARTITION BY symbol ORDER BY year,quarter)) / LAG(revenue)*100 AS change
FROM mart_master_analysis WHERE symbol='VCB' AND year BETWEEN 2023 AND 2024;
```
**RAG Prompt:** “Horizontal analysis IS VCB 2023-2024.”

### **Use Case A8 — Phân tích cơ cấu (Vertical/Common-Size)**
**Mục tiêu nghiệp vụ:** % cấu trúc trong kỳ. Edge: Total=0 → null.

**Input:** Ticker, Period, Report_type ('IS'/'BS').

**Output:** Bảng: item, value, % (e.g., cogs/revenue).

**OLAP/Data Source:** fact_income_statement or fact_balance_sheet.

**Logic/Tính Toán:** % = item / total *100 (e.g., revenue=100%, cogs/cogs*100).

**Example SQL:** 
```sql
SELECT 'revenue' AS item, revenue, 100 AS pct UNION SELECT 'cogs', cogs, cogs/revenue*100 FROM fact_income_statement WHERE symbol='HPG' AND year=2024 AND quarter=4;
```
**RAG Prompt:** “Vertical analysis BS HPG Q4-2024.”

### **Use Case A9 — Phân tích khả năng thanh toán & rủi ro nợ**
**Mục tiêu nghiệp vụ:** Ratios, alerts (D/E>2 high). Edge: Overdraft từ extra_items Map.

**Input:** Ticker, Period.

**Output:** d/e, current_ratio, interest_coverage, kd, short/long ratio; narrative risk.

**OLAP/Data Source:** mart_master_analysis + fact_balance_sheet.

**Logic/Tính Toán:** Kd = interest_expense / total_debt * (1-tax); Ratios pre-calc.

**Example SQL:** 
```sql
SELECT debt_to_equity, current_ratio, interest_coverage, (i.interest_expense / (b.short_term_debt + b.long_term_debt)) * (1 - 0.2) AS kd
FROM mart_master_analysis m JOIN fact_income_statement i ON ... JOIN fact_balance_sheet b ON ... WHERE symbol='VCB' AND year=2024;
```
**RAG Prompt:** “Đánh giá rủi ro nợ, thanh toán VCB 2024, bao Kd, overdraft.”

### **Use Case A10 — Xác định Break-Even / Điểm hòa vốn**
**Mục tiêu nghiệp vụ:** Revenue hòa vốn, sensitivity. Edge: Costs estimate nếu missing.

**Input:** Ticker, Period; fixed/variable costs (input hoặc estimate từ OLAP).

**Output:** break_even = fixed / (1 - variable/revenue); sensitivity (+10% costs).

**OLAP/Data Source:** fact_income_statement (fixed~sgna+interest, variable~cogs).

**Logic/Tính Toán:** Code_execution cho sensitivity; Edge: revenue=0 → null.

**Example SQL:** 
```sql
SELECT (sgna + interest_expense) / (1 - cogs / revenue) AS break_even FROM fact_income_statement WHERE symbol='HPG' AND year=2024;
```
**RAG Prompt:** “Điểm hòa vốn HPG Q4-2024, sensitivity costs tăng 10%.”

### **Use Case A11 — Tạo báo cáo MD&A-style tự động**
**Mục tiêu nghiệp vụ:** Narrative biến động KPI, link macro. Edge: Multi-KPI.

**Input:** Ticker, Period.

**Output:** Text narrative (e.g., "Revenue +15% do mở rộng, margin - do CPI cao").

**OLAP/Data Source:** mart_master_analysis + fact_macro_timeseries (join on year).

**Logic/Tính Toán:** Growth from mart; Corr KPI-macro on-fly.

**Example SQL:** 
```sql
SELECT revenue_growth_yoy, net_margin, (SELECT value FROM fact_macro_timeseries WHERE indicator_code='VN_CPI_YOY' AND toYear(date)=year) AS cpi
FROM mart_master_analysis WHERE symbol='VCB' AND year=2024;
```
**RAG Prompt:** “MD&A VCB 2024, giải thích biến động với macro.”

### **Use Case A12 — Phân tích DuPont**
**Mục tiêu nghiệp vụ:** Breakdown ROE, delta over periods.

**Input:** Ticker, Period.

**Output:** roe, margin, turnover, multiplier; YoY changes; narrative.

**OLAP/Data Source:** mart_master_analysis.dupont_roe.

**Logic/Tính Toán:** roe = margin * turnover * multiplier; Delta = cur - lag.

**Example SQL:** 
```sql
SELECT dupont_roe, net_margin, asset_turnover, equity_multiplier FROM mart_master_analysis WHERE symbol='VCB' AND year=2024;
```
**RAG Prompt:** “DuPont ROE VCB 2024, với changes YoY.”

### **Use Case A13 — Altman Z-Score**
**Mục tiêu nghiệp vụ:** Bankruptcy risk, interpretation. Edge: <1.8 high risk.

**Input:** Ticker, Period.

**Output:** z_score, risk_level (safe/gray/distress), factors breakdown.

**OLAP/Data Source:** mart_master_analysis.altman_z_score.

**Logic/Tính Toán:** Pre-calc; Level: if(z>3,'safe',if(z<1.8,'distress','gray')).

**Example SQL:** 
```sql
SELECT altman_z_score, if(altman_z_score>3,'Safe',if(<1.8,'Distress','Gray')) AS level FROM mart_master_analysis WHERE symbol='HPG' AND year=2024;
```
**RAG Prompt:** “Altman Z-Score HPG, đánh giá rủi ro phá sản.”

### **Use Case A14 — Phân tích Trái phiếu (Bond Analysis)**
**Mục tiêu nghiệp vụ:** YTM/IRR/NPV, risk vs benchmark. Edge: No bonds → note; multiple → list.

**Input:** Ticker, Bond_id (bond_data.bond_id, or all).

**Output:** issuance_date, maturity, coupon, ytm, irr, npv (at r=benchmark); narrative (high YTM → risk).

**OLAP/Data Source:** bond_data (issuance_date, maturity_date, coupon_rate, cash_flows Array), join fact_macro (benchmark=Rf).

**Logic/Tính Toán:** YTM/IRR/NPV code_execution (scipy.optimize.root, numpy.irr/npv); Benchmark = Rf + spread (default 2%).

**Example SQL:** 
```sql
SELECT issuance_date, maturity_date, coupon_rate FROM bond_data WHERE symbol='VCB';
-- Calc: code_execution with def ytm_func(r): npv = sum(cf / (1+r)**t) - price; solve r.
```
**RAG Prompt:** “Phân tích YTM/IRR/NPV trái phiếu VCB 2024, so benchmark.”

### **Use Case A15 — Tính WACC từ Kd/Ke**
**Mục tiêu nghiệp vụ:** Cost of capital, sensitivity. Edge: Forecast via forecast_table.

**Input:** Ticker, Period (forecast_year from forecast_table).

**Output:** wacc, kd, ke; sensitivity (beta ±0.1); narrative.

**OLAP/Data Source:** mart_master_analysis (beta), fact_balance_sheet (E/D), forecast_table (Rm, Rf), fact_macro_timeseries fallback.

**Logic/Tính Toán:** Ke = Rf + beta*(Rm-Rf); WACC as glossary; Sensitivity code_execution loop.

**Example SQL:** 
```sql
SELECT (total_equity / v * ke) + (debt / v * kd * (1-0.2)) AS wacc FROM fact_balance_sheet b JOIN forecast_table f ON b.symbol=f.symbol WHERE b.symbol='HPG';
-- Ke: (SELECT value FROM fact_macro_timeseries WHERE indicator_code='SBV_RATE') + beta * (Rm - Rf).
```
**RAG Prompt:** “WACC HPG từ Kd/Ke forecast, sensitivity.”

### **Use Case A16 — Fraud Detection Analysis**
**Mục tiêu nghiệp vụ:** Detect manipulation via Beneish M-Score. Edge: Score > -2.22 suspicious.

**Input:** Ticker, Period.

**Output:** m_score, components (DSRI etc.); flags; narrative.

**OLAP/Data Source:** mart_master_analysis (growth, margins, assets).

**Logic/Tính Toán:** M-Score = -4.84 + 0.92*DSRI + 0.528*GMI + ... (code_execution full formula); DSRI = (receivables cur / revenue cur) / (prev / prev).

**Example SQL:** 
```sql
SELECT receivables_turnover, asset_growth_yoy FROM mart_master_analysis WHERE symbol='VCB';
-- Calc: code_execution with formula.
```
**RAG Prompt:** “Phân tích gian lận VCB dùng Beneish M-Score.”

## ===========================
## B. Nhà đầu tư / Trader
## ===========================
### **Use Case B1 — Phân tích biến động giá & thị trường**
**Mục tiêu nghiệp vụ:** Volatility, volume, beta. Edge: Custom timeframe.

**Input:** Ticker, Timeframe (e.g., '30d' from fact_daily_market.date).

**Output:** vol_30d, avg_volume, beta; distribution JSON; narrative.

**OLAP/Data Source:** fact_daily_market.

**Logic/Tính Toán:** Vol = stddevPop(close); Beta = cov(stock_ret, index_ret)/var(index_ret) (index từ fact_macro_timeseries 'VN_INDEX').

**Example SQL:** 
```sql
SELECT stddevPop(close) AS vol, avg(volume) AS avg_vol FROM fact_daily_market WHERE symbol='HPG' AND date >= subtractDays(now(),30);
-- Beta: WITH ret AS (...) SELECT covariancePop(stock_ret, index_ret)/variancePop(index_ret) FROM ret JOIN fact_macro_timeseries ON date=date.
```
**RAG Prompt:** “Biến động giá 30d HPG.”

### **Use Case B2 — Dự báo & cảnh báo tín hiệu**
**Mục tiêu nghiệp vụ:** Alerts on KPI/price drops. Edge: Custom rules.

**Input:** Ticker, Rules (e.g., 'eps_drop>10%', 'kd_increase>5%').

**Output:** Alerts list with reasons, periods.

**OLAP/Data Source:** mart_master_analysis + bond_data.

**Logic/Tính Toán:** WHERE conditions on growth/lag; Kd on-fly.

**Example SQL:** 
```sql
SELECT * FROM mart_master_analysis WHERE symbol='VCB' AND eps_growth_yoy < -10;
```
**RAG Prompt:** “Cảnh báo EPS giảm 2 kỳ hoặc Kd tăng >5%.”

### **Use Case B3 — So sánh cổ phiếu với nhóm peer**
**Mục tiêu nghiệp vụ:** Ranking by KPI. Edge: Top N.

**Input:** Sector (dim_company), KPI ('pe_ttm'), Top_n (10).

**Output:** Ranked table: symbol, value; narrative.

**OLAP/Data Source:** mart_master_analysis join dim_company.

**Logic/Tính Toán:** ORDER BY kpi ASC/DESC LIMIT n.

**Example SQL:** 
```sql
SELECT symbol, pe_ttm FROM mart_master_analysis JOIN dim_company ON symbol=symbol WHERE sector='Steel' ORDER BY pe_ttm ASC LIMIT 10;
```
**RAG Prompt:** “Ranking ngành thép theo PE.”

### **Use Case B4 — Phân tích hiệu ứng mùa vụ & TTM chuẩn xác**
**Mục tiêu nghiệp vụ:** TTM KPIs vs annual. Edge: Incomplete → pro-rate.

**Input:** Ticker, Timeframe.

**Output:** revenue_ttm, eps_ttm; comparison; narrative.

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** SUM over last 4 quarters.

**Example SQL:** 
```sql
SELECT sum(revenue) AS ttm FROM mart_master_analysis WHERE symbol='VCB' AND report_date >= subtractQuarters(now(),4);
```
**RAG Prompt:** “TTM revenue VCB loại mùa vụ.”

### **Use Case B5 — So sánh hiệu quả giữa cổ phiếu & benchmark thị trường**
**Mục tiêu nghiệp vụ:** Relative return. Edge: Custom index.

**Input:** Ticker, Timeframe.

**Output:** beta, alpha, relative_return.

**OLAP/Data Source:** fact_daily_market + fact_macro_timeseries (index).

**Logic/Tính Toán:** Beta cov/var; Alpha = stock_ret - (Rf + beta*(index_ret - Rf)).

**Example SQL:** 
```sql
WITH ret AS (SELECT (close - lag(close))/lag(close) AS stock_ret FROM fact_daily_market WHERE symbol='HPG') 
SELECT covariancePop(stock_ret, index_ret)/variancePop(index_ret) AS beta FROM ret JOIN fact_macro_timeseries ON date=date WHERE indicator_code='VN_INDEX';
```
**RAG Prompt:** “So sánh return HPG vs VN-Index 1 năm.”

### **Use Case B6 — Alerts based on financial triggers**
**Mục tiêu nghiệp vụ:** Threshold alerts. Edge: Historical.

**Input:** Ticker, Rules.

**Output:** Triggers list.

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** WHERE net_margin < lag -10 etc.

**Example SQL:** 
```sql
SELECT period, 'Margin drop' FROM mart_master_analysis WHERE symbol='VCB' AND net_margin < lag(net_margin) -10;
```
**RAG Prompt:** “Alerts trigger VCB, bao YTM high.”

### **Use Case B7 — RSI / MACD Signal**
**Mục tiêu nghiệp vụ:** Technical signals. Edge: Custom periods.

**Input:** Ticker, Days (14).

**Output:** rsi, macd; signals (>70 overbought).

**OLAP/Data Source:** fact_daily_market.

**Logic/Tính Toán:** Code_execution (numpy for EMA, gains/losses avg).

**Example SQL:** 
```sql
SELECT close FROM fact_daily_market WHERE symbol='HPG' ORDER BY date DESC LIMIT 30;
-- Code: RS = avg_gain / avg_loss; RSI=100-100/(1+RS).
```
**RAG Prompt:** “RSI, MACD HPG, signals.”

### **Use Case B8 — Đánh giá Trái phiếu cho Đầu tư**
**Mục tiêu nghiệp vụ:** Attractiveness vs benchmark. Edge: Multi-bonds.

**Input:** Ticker, Bond_id.

**Output:** ytm vs benchmark, irr; signals (ytm > benchmark attractive).

**OLAP/Data Source:** bond_data + fact_macro (benchmark).

**Logic/Tính Toán:** YTM code_execution; Compare ytm > Rf +2%.

**Example SQL:** 
```sql
SELECT coupon_rate, current_price FROM bond_data WHERE symbol='HPG';
-- Code for ytm.
```
**RAG Prompt:** “Đánh giá YTM, IRR trái phiếu HPG so lãi suất thị trường.”

## ===========================
## C. CEO / CFO / Quản lý
## ===========================
### **Use Case C1 — Kiểm soát rủi ro tài chính nội bộ**
**Mục tiêu nghiệp vụ:** Anomaly detection. Edge: Threshold custom.

**Input:** Thresholds (e.g., margin_drop>20%).

**Output:** Anomalies list (KPI, period).

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** WHERE net_margin < lag -20 OR short_term_debt >0.5*total_debt.

**Example SQL:** 
```sql
SELECT symbol, year, quarter FROM mart_master_analysis WHERE net_margin < lag(net_margin) -20;
```
**RAG Prompt:** “Anomalies margin giảm >20% hoặc overdraft tăng.”

### **Use Case C2 — So sánh kế hoạch vs thực tế**
**Mục tiêu nghiệp vụ:** Gap analysis. Edge: No budget → note.

**Input:** Ticker, Period (budget_table).

**Output:** actual, plan, %attainment.

**OLAP/Data Source:** mart_master_analysis + budget_table (join period_id).

**Logic/Tính Toán:** % = actual / plan *100.

**Example SQL:** 
```sql
SELECT m.revenue AS actual, bt.revenue AS plan, (actual/plan)*100 FROM mart_master_analysis m JOIN budget_table bt ON m.period_id=bt.period_id WHERE symbol='VCB' AND year=2024;
```
**RAG Prompt:** “Budget vs actual revenue Q4-2024.”

### **Use Case C3 — Phân tích tác động vĩ mô**
**Mục tiêu nghiệp vụ:** Correlation macro-KPI. Edge: Custom indicators.

**Input:** Macro_codes ('VN_CPI_YOY' từ dim_macro_indicator).

**Output:** corr value, trend; narrative.

**OLAP/Data Source:** fact_macro_timeseries join mart_master_analysis on year.

**Logic/Tính Toán:** corr(value, net_margin).

**Example SQL:** 
```sql
SELECT corr(m.value, k.net_margin) FROM fact_macro_timeseries m JOIN mart_master_analysis k ON toYear(m.date)=k.year WHERE m.indicator_code='VN_CPI_YOY';
```
**RAG Prompt:** “Ảnh hưởng CPI đến net margin ngân hàng.”

### **Use Case C4 — Budget vs Actual trend analysis**
**Mục tiêu nghiệp vụ:** Time series gaps.

**Input:** Ticker, Periods.

**Output:** Trend: period, actual, budget, %.

**OLAP/Data Source:** mart_master_analysis + budget_table.

**Logic/Tính Toán:** %attainment per period.

**Example SQL:** 
```sql
SELECT period, actual_revenue, budget_revenue, (actual/budget)*100 FROM mart_master_analysis JOIN budget_table ON period_id=period_id WHERE symbol='VCB' ORDER BY period;
```
**RAG Prompt:** “Trend budget vs actual revenue VCB.”

### **Use Case C5 — Liquidity stress testing**
**Mục tiêu nghiệp vụ:** Scenarios (cash -20%). Edge: Multi-scenarios.

**Input:** Ticker, Scenarios (factors list).

**Output:** Stressed ratios, survival_time (cash / burn_rate).

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** Stressed = ratio * (1 - factor); Code_execution loop.

**Example SQL:** 
```sql
SELECT current_ratio * (1 - 0.2) AS stressed FROM mart_master_analysis WHERE symbol='HPG';
```
**RAG Prompt:** “Stress liquidity HPG cash giảm 20%, lãi suất +10%.”

### **Use Case C6 — Phân tích what-if & kịch bản chiến lược**
**Mục tiêu nghiệp vụ:** Simulate variables. Edge: Probabilistic (monte carlo).

**Input:** Ticker, Variables (e.g., interest +1%).

**Output:** Projected eps, wacc; distributions nếu monte.

**OLAP/Data Source:** mart_master_analysis + forecast_table.

**Logic/Tính Toán:** Projected = base * (1 + impact); Code_execution monte carlo (numpy.random).

**Example SQL:** 
```sql
SELECT eps * (1 + 0.1) AS proj FROM mart_master_analysis WHERE symbol='VCB';
```
**RAG Prompt:** “What-if lãi suất +1% tác động EPS, WACC VCB.”

### **Use Case C7 — Cash Conversion Cycle**
**Mục tiêu nghiệp vụ:** Efficiency, suggestions.

**Input:** Ticker, Period.

**Output:** ccc, DIO/DSO/DPO; optimizations.

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** ccc = days_inventory + days_sales - days_payables.

**Example SQL:** 
```sql
SELECT cash_conversion_cycle FROM mart_master_analysis WHERE symbol='HPG' AND year=2024;
```
**RAG Prompt:** “CCC HPG, đánh giá hiệu quả.”

### **Use Case C8 — Quản lý Cấu trúc Nợ & Lãi suất**
**Mục tiêu nghiệp vụ:** Debt mix, refinance. Edge: Overdraft from extra_items.

**Input:** Ticker, Period.

**Output:** short/long ratio, interest_expense, kd; suggestions.

**OLAP/Data Source:** fact_balance_sheet + fact_income_statement.

**Logic/Tính Toán:** Ratio = short / total_debt.

**Example SQL:** 
```sql
SELECT short_term_debt / (short_term_debt + long_term_debt) AS ratio, interest_expense FROM fact_balance_sheet WHERE symbol='HPG' AND year=2024;
```
**RAG Prompt:** “Cấu trúc nợ ngắn/dài, overdraft, lãi suất HPG.”

### **Use Case C9 — Đánh giá Trái phiếu Nội bộ**
**Mục tiêu nghiệp vụ:** Review issuance, refinancing NPV.

**Input:** Ticker, Bond_id.

**Output:** Metrics, npv refinance; rủi ro.

**OLAP/Data Source:** bond_data.

**Logic/Tính Toán:** NPV at new_rate code_execution.

**Example SQL:** 
```sql
SELECT issuance_date, ytm FROM bond_data WHERE symbol='VCB';
```
**RAG Prompt:** “Đánh giá trái phiếu VCB, NPV refinancing.”

### **Use Case C10 — Compliance and Regulatory Reporting**
**Mục tiêu nghiệp vụ:** Check standards (IFRS). Edge: Custom rules.

**Input:** Ticker, Standard ('IFRS').

**Output:** Compliance score, issues; report narrative.

**OLAP/Data Source:** mart_master_analysis + new compliance_table (rules).

**Logic/Tính Toán:** Score = sum(if(compliant,1,0)) / rules_count.

**Example SQL:** 
```sql
SELECT if(debt_to_equity <2,'Compliant','Non') FROM mart_master_analysis WHERE symbol='VCB';
```
**RAG Prompt:** “Compliance IFRS VCB, generate report.”

### **Use Case C11 — Risk Forecasting with Macro Integration**
**Mục tiêu nghiệp vụ:** Predict under scenarios.

**Input:** Scenario variables (macro changes).

**Output:** Forecasted KPIs; charts JSON.

**OLAP/Data Source:** fact_macro_timeseries + forecast_table.

**Logic/Tính Toán:** Regression code_execution (statsmodels); Forecast = base + corr * delta_macro.

**Example SQL:** 
```sql
SELECT value, net_margin FROM fact_macro_timeseries JOIN mart_master_analysis ON year=year WHERE indicator_code='VN_CPI_YOY';
-- Code for linregress.
```
**RAG Prompt:** “Forecast EPS VCB nếu CPI +5%, dựa corr lịch sử.”

## ===========================
## D. Người dùng cuối / Phổ thông (General User)
## ===========================
### **Use Case D1 — Hỏi tài chính cơ bản**
**Mục tiêu nghiệp vụ:** Định nghĩa đơn giản, ví dụ. Edge: No OLAP.

**Input:** Question (e.g., "Lãi là gì?").

**Output:** Text explanation.

**OLAP/Data Source:** None (static knowledge).

**Logic/Tính Toán:** N/A.

**RAG Prompt:** “Giải thích lãi suất đơn giản.”

### **Use Case D2 — “Công ty này có tốt không?”**
**Mục tiêu nghiệp vụ:** Score dựa data. Edge: Threshold ROE>15 good.

**Input:** Ticker.

**Output:** Score 0-10, reasons (ROE high good, debt high bad).

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** Score = avg(roe/20, 1-debt_to_equity, etc.) *10.

**Example SQL:** 
```sql
SELECT roe_ttm, debt_to_equity FROM mart_master_analysis WHERE symbol='HPG' LIMIT 1;
```
**RAG Prompt:** “HPG có tốt không, dựa data.”

### **Use Case D3 — “Có nên đầu tư cổ phiếu X không?”**
**Mục tiêu nghiệp vụ:** Pros/cons data-based, no advice.

**Input:** Ticker.

**Output:** Pros (growth cao), cons (vol high); narrative.

**OLAP/Data Source:** mart_master_analysis + benchmarks.

**Logic/Tính Toán:** Compare vs sector_avg.

**Example SQL:** 
```sql
SELECT revenue_growth_yoy, volatility_30d FROM mart_master_analysis WHERE symbol='HPG';
```
**RAG Prompt:** “Dựa data, HPG hấp dẫn đầu tư? Pros/cons.”

### **Use Case D4 — Giải thích KPI tài chính theo ngôn ngữ đơn giản**
**Mục tiêu nghiệp vụ:** Layman explain + value. Edge: With example.

**Input:** Ticker, KPI ('ROE').

**Output:** "ROE 15% nghĩa là kiếm 15đ từ 100đ vốn"; value.

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** Fetch value.

**Example SQL:** 
```sql
SELECT roe_ttm FROM mart_master_analysis WHERE symbol='VCB';
```
**RAG Prompt:** “Giải thích ROE VCB đơn giản.”

### **Use Case D5 — So sánh hiệu quả kinh doanh giữa hai công ty bất kỳ**
**Mục tiêu nghiệp vụ:** Quick compare. Edge: >2 tickers.

**Input:** Tickers list, Period.

**Output:** Table: symbol, revenue, margin, roe.

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** UNION or IN.

**Example SQL:** 
```sql
SELECT symbol, revenue, net_margin, roe_ttm FROM mart_master_analysis WHERE symbol IN ('VCB','HPG') AND year=2024;
```
**RAG Prompt:** “So sánh VCB vs HPG.”

### **Use Case D6 — Hỏi về rủi ro tài chính tổng quát**
**Mục tiêu nghiệp vụ:** Explain ratios. Edge: Type-specific.

**Input:** Ticker, Risk_type ('debt').

**Output:** Ratios, why (D/E cao → rủi ro lớn).

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** Fetch + threshold check.

**Example SQL:** 
```sql
SELECT debt_to_equity FROM mart_master_analysis WHERE symbol='HPG';
```
**RAG Prompt:** “Rủi ro tài chính HPG cao không? Tại sao, bao overdraft.”

### **Use Case D7 — Peer Comparison (Radar Chart)**
**Mục tiêu nghiệp vụ:** Visual strengths. Edge: Normalize 0-100.

**Input:** Sector or tickers.

**Output:** JSON scores (roe, pe, growth); narrative.

**OLAP/Data Source:** mart_master_analysis.

**Logic/Tính Toán:** Score = (value - min)/(max - min)*100 per KPI, GROUP BY sector.

**Example SQL:** 
```sql
SELECT symbol, roe_ttm, pe_ttm, revenue_growth_yoy FROM mart_master_analysis WHERE sector='Banking';
-- Code normalize.
```
**RAG Prompt:** “Peer comparison VCB ngân hàng với radar.”

### **Use Case D8 — Giải thích Cash Flow Phân loại**
**Mục tiêu nghiệp vụ:** Simple breakdown.

**Input:** Ticker, Period.

**Output:** CFO/CFI/CFF values, explain (CFO dương → hoạt động tốt).

**OLAP/Data Source:** fact_cash_flow.

**Logic/Tính Toán:** Fetch + sign analysis.

**Example SQL:** 
```sql
SELECT cfo, cfi, cff FROM fact_cash_flow WHERE symbol='VCB' AND year=2024;
```
**RAG Prompt:** “Giải thích cash flow CFO/CFI/CFF VCB dễ hiểu.”

### **Use Case D9 — Interactive Q&A with Visuals**
**Mục tiêu nghiệp vụ:** Explain with images/charts. Edge: Visualizable queries.

**Input:** Question.

**Output:** Text + rendered images (via search_images tool in RAG chain).

**OLAP/Data Source:** Optional mart cho data, images từ tool.

**Logic/Tính Toán:** N/A, tool-based.

**RAG Prompt:** “Giải thích ROE đơn giản, với image minh họa.”

## 🧠 Tích hợp vào hệ thống RAG
- Retrieval: SQL từ OLAP → code_execution calc → narrative.
- Pipeline: ETL → mart → RAG chain.
