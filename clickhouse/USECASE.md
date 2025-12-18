# 📌 ENRICHED FINANCIAL ANALYSIS RAG USE CASE SPEC

Tài liệu này mô tả thiết kế các use case cho hệ thống RAG (Retrieval-Augmented Generation) và data pipeline hỗ trợ hỏi đáp phân tích tài chính doanh nghiệp. Hệ thống kết hợp dữ liệu OLAP (từ các mart như mart_master_analysis, fact_daily_market, fact_macro_timeseries), thuật toán tính toán KPI, và generation narrative dựa trên RAG để cung cấp insight sâu sắc, bao gồm tóm tắt, so sánh, định giá, và phân tích rủi ro. Các use case được phân loại theo persona người dùng (Analyst, Trader/Investor, CEO/CFO/Manager, General User), với tích hợp các mở rộng để hỗ trợ phân tích nâng cao như horizontal/vertical analysis, what-if scenarios, và narrative MD&A-style.

Hệ thống không chỉ trả về dữ liệu thô mà còn diễn giải biến động KPI trong bối cảnh kinh tế/doanh nghiệp, tránh tư vấn tài chính pháp lý, và tập trung vào dữ liệu khách quan. Các prompt RAG được thiết kế để kết hợp retrieval từ OLAP với generation tự nhiên.

## 📊 Glossary (Chỉ số KPI)
| KPI | Công thức | Mô tả |
|-----|-----------|-------|
| PE TTM | price / (eps*4) | Định giá dựa trên lợi nhuận trailing twelve months. |
| PB | price / bvps | Định giá dựa trên giá trị sổ sách. |
| ROE | NI*4 / equity*100 | Hiệu quả sinh lời trên vốn chủ sở hữu. |
| FCF Yield | FCF*4 / market_cap*100 | Lợi suất dòng tiền tự do. |
| Volatility | STDDEV(close) | Độ biến động giá cổ phiếu. |
| ROA TTM | NI*4 / total_assets*100 | Hiệu quả sinh lời trên tài sản. |
| ROIC | NOPAT / invested_capital | Hiệu quả sinh lời trên vốn đầu tư. |
| EV/EBITDA | (market_cap + debt - cash) / EBITDA | Định giá doanh nghiệp. |
| Dividend Yield | dividends / price * 100 | Lợi suất cổ tức. |
| Growth YoY | (current - previous) / previous * 100 | Tăng trưởng năm trên năm. |
| CFO/Revenue | Cash from Operations / Revenue | Chất lượng dòng tiền hoạt động. |
| Accrual Ratio | (NI - CFO) / total_assets | Chất lượng lợi nhuận (thấp hơn tốt hơn). |
| Current Ratio | Current Assets / Current Liabilities | Khả năng thanh toán ngắn hạn. |
| D/E | Debt / Equity | Rủi ro nợ. |
| Cash Conversion Cycle | DIO + DSO - DPO | Hiệu quả vốn lưu động (Days Inventory Outstanding + Days Sales Outstanding - Days Payable Outstanding). |
| Altman Z-Score | 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MV/BV) + 1.0*(Sales/TA) | Dự báo rủi ro phá sản. |
| RSI | 100 - (100 / (1 + RS)), RS = Avg Gain / Avg Loss | Chỉ báo kỹ thuật quá mua/quá bán. |
| MACD | EMA12 - EMA26 | Chỉ báo xu hướng động lượng. |

## ===========================
## A. Người phân tích / Chuyên gia tài chính (Analyst)
## ===========================
Các use case tập trung vào phân tích sâu, sử dụng OLAP để tổng hợp KPI, trend, và insight narrative. Bao gồm các kỹ thuật căn bản như horizontal/vertical analysis, DuPont, và MD&A-style generation.

### **Use Case 1 — Tóm tắt tình hình tài chính doanh nghiệp**
**Mục tiêu nghiệp vụ**  
Cung cấp summary dashboard toàn diện về tình hình tài chính hiện tại của một công ty theo kỳ, kết hợp số liệu IS/BS/CF & market.  

**Input**  
Ticker (ví dụ: VCB), Kỳ tài chính (Q4-2024, YTD, TTM).  

**Output**  
Bảng chỉ số tóm tắt: revenue, net income, EPS, ROE ttm, ROA ttm, cash flow chất lượng, margin, market cap.  

**Data OLAP**  
mart_master_analysis (core KPIs), fact_daily_market (giá & volume).  

**Thuật toán/Tính toán**  
Tổng hợp KPI từ mart_master_analysis, lấy giá và volume gần nhất.  

**Example queries**  
```sql
SELECT * FROM mart_master_analysis WHERE symbol='VCB' AND year=2024 AND quarter=4;
```  

**Example RAG Prompt**  
> “Tóm tắt tình hình tài chính VCB đến Q4 2024 với các chỉ số chính.”

### **Use Case 2 — So sánh chỉ số tài chính giữa các kỳ**
**Mục tiêu nghiệp vụ**  
Trend analysis: phân tích tăng/giảm YoY/TTM, pattern thay đổi across các chỉ số quan trọng.  

**Input**  
Ticker, list periods (ví dụ Q3-2023, Q4-2023, Q1-2024, Q2-2024).  

**Output**  
Growth % cho revenue, net_income, eps, margin.  

**OLAP**  
mart_master_analysis.  

**Công thức**  
growth_yoy = (current - previous) / previous * 100.  

**Example**  
```sql
SELECT period_label, revenue_growth_yoy, net_margin FROM mart_master_analysis WHERE symbol='HPG' ORDER BY year, quarter;
```  

**Example RAG Prompt**  
> “So sánh doanh thu và biên lợi nhuận của HPG qua 4 quý gần nhất.”

### **Use Case 3 — Phân tích hiệu quả hoạt động**
**Mục tiêu nghiệp vụ**  
Đánh giá hiệu quả sinh lời (ROE, ROA, ROIC), nhận diện điểm mạnh/điểm yếu hoạt động.  

**Input**  
Ticker, target period.  

**Output**  
Giá trị ROE_ttm, ROA_ttm, ROIC + phân tích yếu tố driver.  

**OLAP**  
mart_master_analysis.  

**Example**  
```sql
SELECT roe_ttm, roa_ttm, roic FROM mart_master_analysis WHERE symbol='VCB' AND year=2024 AND quarter=4;
```  

**Example RAG Prompt**  
> “Phân tích hiệu quả hoạt động của VCB dựa trên ROE, ROA và ROIC trong năm 2024.”

### **Use Case 4 — Định giá doanh nghiệp (Valuation)**
**Mục tiêu nghiệp vụ**  
Giá trị định giá: PE, PB, EV/EBITDA, dividend yield; so sánh với mức industry benchmark.  

**Input**  
Ticker & peer_sector, Period.  

**Output**  
Valuation multiples, deviation +/- so với trung vị ngành.  

**OLAP**  
mart_master_analysis, xây thêm bảng sector benchmark.  

**Công thức**  
pe_ttm = price / (eps*4), pb = price / bvps.  

**Example**  
```sql
SELECT pe_ttm, pb, ev_ebitda FROM mart_master_analysis WHERE symbol='HPG';
```  

**RAG Prompt**  
> “Định giá HPG hiện tại so với trung vị ngành thép.”

### **Use Case 5 — Benchmark so với ngành/sector**
**Mục tiêu nghiệp vụ**  
So sánh KPI của một công ty với trung vị ngành/framework peer, đánh giá strengths/weaknesses.  

**Input**  
Ticker, sector, Period.  

**Output**  
Sector average PE/PB/ROE/Growth.  

**OLAP**  
mart_master_analysis aggregated by sector.  

**SQL pattern**  
```sql
SELECT avg(pe_ttm) as avg_pe, avg(roe_ttm) as avg_roe FROM mart_master_analysis WHERE sector='Banking';
```  

**RAG Prompt**  
> “Benchmark ROE và PE của VCB so với các ngân hàng cùng ngành.”

### **Use Case 6 — Phân tích dòng tiền & chất lượng lợi nhuận**
**Mục tiêu nghiệp vụ**  
Đánh giá cash flow quality, FCF yield, CFO/Revenue, accrual ratio.  

**Input**  
Ticker, period.  

**Output**  
CFO/Revenue, FCF/Revenue, accrual ratios.  

**OLAP**  
mart_master_analysis.  

**RAG Prompt**  
> “Phân tích dòng tiền và chất lượng lợi nhuận của VCB trong 2024.”

### **Use Case A16 — Phân tích hoạt động theo phương pháp ngang (Horizontal Analysis)**
**Mục tiêu nghiệp vụ**  
So sánh các chỉ tiêu tài chính qua nhiều kỳ để xác định trend & bất thường.  

**Input**  
Ticker, multi-periods (ví dụ: Q1-2023 to Q4-2024).  

**Output**  
Biểu đồ biến động và % thay đổi YoY cho revenue, expenses, NI; nhận diện bất thường (e.g., spike/drop >20%).  

**OLAP**  
mart_master_analysis theo multi-period và biểu đồ biến động.  

**Công thức**  
% change = (current - base) / base * 100 (base là kỳ đầu tiên).  

**Example SQL**  
```sql
SELECT year, quarter, revenue, (revenue - LAG(revenue) OVER (ORDER BY year, quarter)) / LAG(revenue) OVER (ORDER BY year, quarter) * 100 AS revenue_change FROM mart_master_analysis WHERE symbol='VCB';
```  

**RAG Prompt**  
> “Phân tích horizontal analysis cho IS của VCB từ 2023 đến 2024.”

### **Use Case A17 — Phân tích cơ cấu (Vertical/Common-Size)**
**Mục tiêu nghiệp vụ**  
Phân tích tỉ trọng các khoản mục IS/BS trong cùng một kỳ.  

**Input**  
Ticker, period.  

**Output**  
% từng line item so với tổng (e.g., COGS/revenue, assets/liabilities).  

**OLAP**  
Thống kê % từng line item so với tổng (common-size IS/BS) từ mart_master_analysis.  

**Công thức**  
% = item / total * 100.  

**Example SQL**  
```sql
SELECT item_name, (value / total_revenue) * 100 AS percentage FROM mart_master_analysis WHERE symbol='HPG' AND period='Q4-2024' AND report_type='IS';
```  

**RAG Prompt**  
> “Phân tích vertical analysis cho BS của HPG trong Q4 2024.”

### **Use Case A18 — Phân tích khả năng thanh toán & rủi ro nợ**
**Mục tiêu nghiệp vụ**  
Đánh giá năng lực trả nợ ngắn hạn/dài hạn.  

**Input**  
Ticker, period.  

**Output**  
D/E, current ratio, cash ratio; insight về rủi ro (e.g., nếu D/E >2 thì cao rủi ro).  

**OLAP**  
D/E, current ratio, cash ratio từ mart_master_analysis.  

**Công thức**  
Current Ratio = Current Assets / Current Liabilities.  

**Example SQL**  
```sql
SELECT debt_to_equity, current_ratio FROM mart_master_analysis WHERE symbol='VCB' AND year=2024;
```  

**RAG Prompt**  
> “Đánh giá rủi ro nợ và khả năng thanh toán của VCB năm 2024.”

### **Use Case A19 — Xác định Break-Even / Điểm hòa vốn**
**Mục tiêu nghiệp vụ**  
Tính điểm doanh thu tối thiểu để hòa vốn.  

**Input**  
Ticker, period; chi phí cố định/biến đổi (từ OLAP hoặc external).  

**Output**  
Break-even revenue = fixed_costs / (1 - variable_costs/revenue).  

**OLAP**  
Dữ liệu chi phí cố định/biến đổi từ mart_master_analysis hoặc external nhập.  

**Công thức**  
Break-even = Fixed Costs / Contribution Margin Ratio.  

**Example SQL**  
```sql
SELECT fixed_costs / (1 - variable_costs / revenue) AS break_even FROM mart_master_analysis WHERE symbol='HPG';
```  

**RAG Prompt**  
> “Điểm hòa vốn của HPG trong Q4 2024 là bao nhiêu?”

### **Use Case A20 — Tạo báo cáo MD&A-style tự động**
**Mục tiêu nghiệp vụ**  
Giải thích narrative & lý do đằng sau biến động KPI.  

**Input**  
Ticker, period.  

**Output**  
Narrative text (e.g., "Revenue tăng 15% do mở rộng thị trường, nhưng margin giảm vì chi phí nguyên liệu cao.").  

**OLAP**  
Kết hợp dữ liệu trend từ mart_master_analysis + NLP generation.  

**Example SQL**  
```sql
SELECT revenue_growth_yoy, net_margin_change FROM mart_master_analysis WHERE symbol='VCB';
```  

**RAG Prompt**  
> “Tạo báo cáo MD&A cho VCB năm 2024, giải thích biến động KPI.”

### **Use Case A21 — Phân tích DuPont**
**Mục tiêu nghiệp vụ**  
Giải thích sâu nguyên nhân biến động ROE qua các driver (margin, turnover, leverage).  

**Input**  
Ticker, period (IS, BS).  

**Output**  
ROE = Net Margin * Asset Turnover * Equity Multiplier; phân tích thay đổi từng yếu tố.  

**OLAP**  
mart_master_analysis (NI, revenue, assets, equity).  

**Công thức**  
ROE = (NI/Revenue) * (Revenue/Assets) * (Assets/Equity).  

**Example SQL**  
```sql
SELECT (net_income/revenue) * (revenue/total_assets) * (total_assets/equity) AS roe_dupont FROM mart_master_analysis WHERE symbol='VCB';
```  

**RAG Prompt**  
> “Phân tích DuPont cho ROE của VCB năm 2024.”

### **Use Case A22 — Altman Z-Score**
**Mục tiêu nghiệp vụ**  
Đánh giá nguy cơ phá sản / kiệt quệ tài chính.  

**Input**  
Ticker, period (BS, Market Data).  

**Output**  
Z-Score value; interpretation (e.g., >3: an toàn, <1.8: rủi ro cao).  

**OLAP**  
mart_master_analysis + fact_daily_market.  

**Công thức**  
Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MV/BV) + 1.0*(Sales/TA).  

**Example SQL**  
```sql
SELECT 1.2*(working_capital/total_assets) + ... FROM mart_master_analysis WHERE symbol='HPG';
```  

**RAG Prompt**  
> “Tính Altman Z-Score cho HPG và đánh giá rủi ro phá sản.”

## ===========================
## B. Nhà đầu tư / Trader
## ===========================
Tập trung vào phân tích thị trường, tín hiệu, và so sánh để hỗ trợ quyết định đầu tư/trading, với tích hợp TTM và technical indicators.

### **Use Case 7 — Phân tích biến động giá & thị trường**
**Mục tiêu nghiệp vụ**  
Tính volatility 30/60/90 days, market beta nếu có benchmark.  

**Input**  
Ticker, timeframe.  

**Output**  
Volatility, avg volume, return distribution.  

**OLAP**  
fact_daily_market.  

**SQL**  
```sql
SELECT STDDEV(close) AS vol_30d, AVG(volume) AS avg_volume FROM fact_daily_market WHERE symbol='HPG' AND date>= today()-30;
```  

**RAG Prompt**  
> “Tính độ biến động giá 30 ngày gần nhất của HPG.”

### **Use Case 8 — Dự báo & cảnh báo tín hiệu**
**Mục tiêu nghiệp vụ**  
Cảnh báo giảm mạnh KPI/price trend, phát hiện bất thường.  

**Input**  
Ticker, alert rules.  

**Output**  
Alert signals: price crash, margin drop > X%.  

**OLAP**  
mart_master_analysis + fact_daily_market series.  

**RAG Prompt**  
> “Cảnh báo nếu EPS giảm liên tục 2 kỳ.”

### **Use Case 9 — So sánh cổ phiếu với nhóm peer**
**Mục tiêu nghiệp vụ**  
Ranking theo PE/PB/Growth trong ngành.  

**Input**  
Sector.  

**Output**  
Ranked list.  

**SQL**  
```sql
SELECT symbol, pe_ttm, revenue_growth_yoy FROM mart_master_analysis WHERE sector='Steel' ORDER BY pe_ttm ASC;
```  

**RAG Prompt**  
> “Xếp hạng cổ phiếu ngành thép theo PE.”

### **Use Case B10 — Phân tích hiệu ứng mùa vụ & TTM chuẩn xác**
**Mục tiêu nghiệp vụ**  
Đánh giá performance liên tục 12 tháng gần nhất, loại bỏ mùa vụ.  

**Input**  
Ticker, timeframe.  

**Output**  
TTM KPIs (e.g., revenue_ttm, eps_ttm); so sánh với annual.  

**OLAP**  
Thống kê TTM từ fact IS / mart layer.  

**Công thức**  
TTM = sum(last 4 quarters).  

**Example SQL**  
```sql
SELECT SUM(revenue) AS revenue_ttm FROM mart_master_analysis WHERE symbol='VCB' AND period IN (last_4_quarters);
```  

**RAG Prompt**  
> “Phân tích TTM revenue của VCB để loại bỏ mùa vụ.”

### **Use Case B11 — So sánh hiệu quả giữa cổ phiếu & benchmark thị trường**
**Mục tiêu nghiệp vụ**  
So sánh return, volatility vs VN-Index.  

**Input**  
Ticker, timeframe.  

**Output**  
Beta, alpha, relative return.  

**OLAP**  
fact_daily_market + index data.  

**Công thức**  
Beta = COV(stock_return, index_return) / VAR(index_return).  

**Example SQL**  
```sql
SELECT COVARIANCE(stock_close, index_close) / VARIANCE(index_close) AS beta FROM fact_daily_market WHERE symbol='HPG';
```  

**RAG Prompt**  
> “So sánh return của HPG với VN-Index trong 1 năm.”

### **Use Case B12 — Alerts based on financial triggers**
**Mục tiêu nghiệp vụ**  
Tự động cảnh báo khi KPI vượt threshold.  

**Input**  
Ticker, rules (e.g., margin drop >10%).  

**Output**  
List alerts với lý do.  

**OLAP**  
mart_master_analysis trend + rules engine.  

**Example SQL**  
```sql
SELECT * FROM mart_master_analysis WHERE symbol='VCB' AND net_margin_change < -10;
```  

**RAG Prompt**  
> “Cảnh báo các trigger tài chính cho VCB dựa trên rules.”

### **Use Case B13 — RSI / MACD Signal**
**Mục tiêu nghiệp vụ**  
Tín hiệu quá mua / quá bán (Technical).  

**Input**  
Ticker, daily price.  

**Output**  
RSI value, MACD line; signals (e.g., RSI>70: overbought).  

**OLAP**  
fact_daily_market.  

**Công thức**  
RSI = 100 - (100 / (1 + RS)); MACD = EMA12 - EMA26.  

**Example SQL**  
```sql
-- Sử dụng code execution để tính EMA/RSI nếu cần.
SELECT close FROM fact_daily_market WHERE symbol='HPG' ORDER BY date DESC LIMIT 30;
```  

**RAG Prompt**  
> “Tính RSI và MACD cho HPG, phát hiện signals.”

## ===========================
## C. CEO / CFO / Quản lý
## ===========================
Tập trung vào kiểm soát nội bộ, planning, và scenario analysis để hỗ trợ quản lý rủi ro và chiến lược.

### **Use Case 10 — Kiểm soát rủi ro tài chính nội bộ**
**Mục tiêu nghiệp vụ**  
Detect KPI anomalies (e.g. net income % drop > threshold, margin compression).  

**Input**  
Threshold rules.  

**Output**  
Anomaly list.  

**Data**  
mart_master_analysis.  

**RAG Prompt**  
> “Liệt kê các công ty có net margin giảm > 20% năm qua.”

### **Use Case 11 — So sánh kế hoạch vs thực tế**
**Mục tiêu nghiệp vụ**  
Plan vs actual KPI comparison.  

**Input**  
Plan data source (external table).  

**Output**  
Gap analysis, % attainment.  

**Data**  
mart_master_analysis + plan table.  

**RAG Prompt**  
> “So sánh doanh thu thực tế và kế hoạch Q4 2024.”

### **Use Case 12 — Phân tích tác động vĩ mô**
**Mục tiêu nghiệp vụ**  
Hiệu ứng CPI, lãi suất xuống EPS, margin.  

**Input**  
Macro timeline.  

**Output**  
Correlation/trend.  

**Data**  
fact_macro_timeseries + mart_master_analysis.  

**SQL**  
```sql
SELECT m.value AS CPI, avg(kpi.net_margin) AS avg_margin FROM fact_macro_timeseries m JOIN mart_master_analysis kpi ON year(m.date)=kpi.year WHERE m.indicator_code='VN_CPI_YOY' GROUP BY year(m.date);
```  

**RAG Prompt**  
> “Ảnh hưởng CPI YoY đến net margin ngành ngân hàng.”

### **Use Case C13 — Budget vs Actual trend analysis**
**Mục tiêu nghiệp vụ**  
So sánh ngân sách kế hoạch với thực tế theo time series.  

**Input**  
Ticker, periods.  

**Output**  
Trend gap % over time.  

**OLAP**  
Budget table + mart_master_analysis.  

**Công thức**  
% Attainment = actual / budget * 100.  

**Example SQL**  
```sql
SELECT period, actual_revenue, budget_revenue, (actual_revenue / budget_revenue * 100) AS attainment FROM mart_master_analysis JOIN budget_table ON period=budget_period WHERE symbol='VCB';
```  

**RAG Prompt**  
> “Phân tích trend budget vs actual cho revenue của VCB.”

### **Use Case C14 — Liquidity stress testing**
**Mục tiêu nghiệp vụ**  
Đánh giá hoạt động trong stress scenarios (e.g., cash drop 20%).  

**Input**  
Ticker, scenarios.  

**Output**  
Projected ratios under stress; survival time.  

**OLAP**  
Liquidity ratios + variance scenarios từ mart_master_analysis.  

**Công thức**  
Stressed Current Ratio = (current_assets * (1 - stress_factor)) / current_liabilities.  

**Example SQL**  
```sql
SELECT current_ratio * (1 - 0.2) AS stressed_ratio FROM mart_master_analysis WHERE symbol='HPG';
```  

**RAG Prompt**  
> “Stress test liquidity cho HPG nếu cash giảm 20%.”

### **Use Case C15 — Phân tích what-if & kịch bản chiến lược**
**Mục tiêu nghiệp vụ**  
Mô phỏng tác động thay đổi biến đầu vào (e.g., lãi suất tăng).  

**Input**  
Ticker, scenario variables.  

**Output**  
Projected KPIs (e.g., new EPS).  

**OLAP**  
Scenario data + OLAP results từ mart_master_analysis.  

**Công thức**  
What-if EPS = base_eps * (1 + impact_factor).  

**Example SQL**  
```sql
-- Sử dụng code execution cho simulation nếu cần.
SELECT eps * (1 + 0.1) AS projected_eps FROM mart_master_analysis WHERE symbol='VCB';
```  

**RAG Prompt**  
> “What-if analysis: Nếu lãi suất tăng 1%, tác động đến EPS của VCB.”

### **Use Case C16 — Cash Conversion Cycle**
**Mục tiêu nghiệp vụ**  
Hiệu quả quản trị vốn lưu động.  

**Input**  
Ticker, period (IS, BS).  

**Output**  
CCC value; optimization suggestions.  

**OLAP**  
mart_master_analysis.  

**Công thức**  
CCC = DIO + DSO - DPO.  

**Example SQL**  
```sql
SELECT days_inventory + days_sales - days_payable AS ccc FROM mart_master_analysis WHERE symbol='HPG';
```  

**RAG Prompt**  
> “Tính Cash Conversion Cycle cho HPG và đánh giá hiệu quả.”

## ===========================
## D. Người dùng cuối / Phổ thông (General User)
## ===========================
Cung cấp giải thích đơn giản, không kỹ thuật, dựa trên dữ liệu nhưng dễ hiểu.

### **Use Case 13 — Hỏi tài chính cơ bản**
**Mục tiêu nghiệp vụ**  
Giải thích khái niệm: ví dụ “Lãi là gì?” Không cần dữ liệu OLAP, chỉ định nghĩa.  

**Input**  
Plain question.  

**Output**  
Natural language explanation.  

**RAG Prompt**  
> “Lãi là gì? Giải thích đơn giản.”

### **Use Case 14 — “Công ty này có tốt không?”**
**Mục tiêu nghiệp vụ**  
Simple quality score, tổng hợp ROE, ROA, margin.  

**OLAP**  
mart_master_analysis.  

**Example RAG Prompt**  
> “Cho tôi biết HPG có phải công ty tốt không.”

### **Use Case 15 — “Có nên đầu tư cổ phiếu X không?”**
**Mục tiêu nghiệp vụ**  
Synthesis data + opinion (không tư vấn pháp lý, chỉ dựa dữ liệu).  

**OLAP**  
mart_master_analysis + benchmarks.  

**RAG Prompt**  
> “Dựa trên dữ liệu, liệu HPG có hấp dẫn để đầu tư?”

### **Use Case D16 — Giải thích KPI tài chính theo ngôn ngữ đơn giản**
**Mục tiêu nghiệp vụ**  
Convert KPI ra lời giải thích dễ hiểu.  

**Input**  
Ticker, KPI (e.g., ROE).  

**Output**  
Explanation + value (e.g., "ROE 15% nghĩa là công ty kiếm 15 đồng lợi nhuận từ mỗi 100 đồng vốn.").  

**OLAP**  
mart_master_analysis + template explanations.  

**Example SQL**  
```sql
SELECT roe_ttm FROM mart_master_analysis WHERE symbol='VCB';
```  

**RAG Prompt**  
> “Giải thích ROE của VCB một cách đơn giản.”

### **Use Case D17 — So sánh hiệu quả kinh doanh giữa hai công ty bất kỳ**
**Mục tiêu nghiệp vụ**  
Cung cấp công cụ so sánh nhanh.  

**Input**  
Hai tickers, period.  

**Output**  
Bảng so sánh KPIs (revenue, margin, ROE).  

**OLAP**  
mart_master_analysis hai ticker.  

**Example SQL**  
```sql
SELECT symbol, revenue, net_margin FROM mart_master_analysis WHERE symbol IN ('VCB', 'HPG') AND year=2024;
```  

**RAG Prompt**  
> “So sánh hiệu quả kinh doanh giữa VCB và HPG.”

### **Use Case D18 — Hỏi về rủi ro tài chính tổng quát**
**Mục tiêu nghiệp vụ**  
Trả lời câu hỏi rủi ro (liquidity, leverage).  

**Input**  
Ticker, rủi ro type.  

**Output**  
Explanation + ratios (e.g., "D/E cao cho thấy rủi ro nợ lớn.").  

**OLAP**  
mart_master_analysis ratios.  

**Example SQL**  
```sql
SELECT debt_to_equity FROM mart_master_analysis WHERE symbol='HPG';
```  

**RAG Prompt**  
> “Công ty HPG có rủi ro tài chính cao không? Tại sao?”

### **Use Case D19 — Peer Comparison (Radar Chart)**
**Mục tiêu nghiệp vụ**  
So sánh sức mạnh tương quan trên nhiều trục (Sinh lời, Định giá, Tăng trưởng).  

**Input**  
Sector hoặc list tickers.  

**Output**  
Radar chart data (scores trên axes); narrative comparison.  

**OLAP**  
Sector Data từ mart_master_analysis.  

**Công thức**  
Normalize scores 0-100 cho từng KPI.  

**Example SQL**  
```sql
SELECT symbol, roe_ttm, pe_ttm, growth_yoy FROM mart_master_analysis WHERE sector='Banking';
```  

**RAG Prompt**  
> “So sánh peer comparison cho VCB trong ngành ngân hàng với radar chart.”

## 🧠 Tích hợp vào hệ thống RAG: Analyst-style Narrative Generation
Hệ thống RAG không chỉ retrieval dữ liệu mà còn generate narrative mô phỏng MD&A (Management Discussion & Analysis), giải thích biến động KPI với lý do kinh tế/doanh nghiệp. Sử dụng prompt chaining: retrieval OLAP → analyze trends → generate explanation.  

**Ví dụ câu hỏi RAG của người dùng:**  
- “Tại sao lợi nhuận gộp của VCB tăng nhưng ROE lại giảm?” (Phân tích driver như leverage giảm.)  
- “Điểm hòa vốn của công ty A trong Q1-Q3 2025 là bao nhiêu?” (Tính toán + giải thích.)  
- “Tôi muốn xem dự báo EPS cho 2 quý tiếp theo dựa trên trend lịch sử.” (Sử dụng linear regression từ data.)  

Data pipeline hỗ trợ: ETL từ nguồn tài chính (BCTC, market data) vào OLAP marts, với rules engine cho alerts/scenarios.

## 🚀 Summary: Nhóm Use Case Bổ Sung Quan Trọng
| Nhóm | Tính năng bổ sung | Lợi ích chính |
|------|-------------------|---------------|
| Analyst | Horizontal analysis, Common-size, Break-Even, Narrative MD&A, DuPont, Altman Z-Score | Insight sâu hơn về trend, cấu trúc, và rủi ro phá sản. |
| Trader/Investor | TTM & benchmark vs index, Alerts, RSI/MACD | Loại bỏ mùa vụ, tín hiệu trading, so sánh thị trường. |
| CEO/CFO/Manager | Budget vs Actual, Stress testing, What-if modelling, Cash Conversion Cycle | Hỗ trợ planning, rủi ro nội bộ, và tối ưu vốn. |
| General User | Explanation in layman terms, Comparative analysis, Radar Chart | Dễ tiếp cận, so sánh trực quan cho người mới. |