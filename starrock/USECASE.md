# 📌 TÀI LIỆU PHÂN TÍCH TÀI CHÍNH DOANH NGHIỆP 
Tài liệu này mô tả chi tiết thiết kế các use case cho hệ thống RAG (Retrieval-Augmented Generation) hỗ trợ phân tích tài chính doanh nghiệp. Hệ thống kết hợp dữ liệu OLAP từ ClickHouse (các bảng như `dim_company, dim_period, fact_income_statement, fact_balance_sheet, fact_cash_flow, fact_daily_market, dim_macro_indicator, fact_macro_timeseries, mart_master_analysis`, và các bảng bổ sung như `bond_data, forecast_table, budget_table`), thuật toán tính toán KPI (pre-calculated trong `mart_master_analysis` qua materialized view `mv_master_analysis`, hoặc on-fly qua `code_execution`), và generation narrative dựa trên RAG. Các use case được phân loại theo persona người dùng (`Analyst, Trader/Investor, CEO/CFO/Manager, General User`), với chi tiết input/output, logic tính toán dựa trên tables, edge cases, và RAG prompt. Hệ thống tránh tư vấn pháp lý, tập trung dữ liệu khách quan.
## 📊 Bảng Thuật Ngữ KPI (Glossary)

| KPI | Công thức | Mô tả | Database Link & Logic |
|------|-----------|--------|------------------------|
| **PE TTM** | price / (eps * 4) | Định giá dựa trên lợi nhuận 12 tháng gần nhất | fact_daily_market.close, fact_income_statement.eps (cộng 4 quý) → mart_master_analysis.pe_ttm |
| **PB** | price / bvps | Định giá theo giá trị sổ sách | fact_daily_market.close, fact_balance_sheet.bvps → mart_master_analysis.pb |
| **ROE TTM** | (NI * 4) / equity * 100 | Tỷ suất sinh lời trên vốn chủ sở hữu (12T) | fact_income_statement.net_income, fact_balance_sheet.total_equity → mart_master_analysis.roe_ttm |
| **ROA TTM** | (NI * 4) / total_assets * 100 | Tỷ suất sinh lời trên tổng tài sản (12T) | net_income, total_assets → mart_master_analysis.roa_ttm |
| **ROIC** | NOPAT / invested_capital | Hiệu quả trên vốn đầu tư thực tế | (operating_profit * (1 − tax_rate)) / (total_assets − current_liabilities) → mart_master_analysis.roic |
| **FCF Yield** | (FCF * 4) / market_cap * 100 | Lợi suất dòng tiền tự do | fact_cash_flow.fcf, fact_daily_market.market_cap → mart_master_analysis.fcf_yield |
| **EV/EBITDA** | (market_cap + debt − cash) / EBITDA | Định giá doanh nghiệp | market_cap + short/long debt + cash + EBITDA → mart_master_analysis.ev_ebitda |
| **Dividend Yield** | dividends / price * 100 | Lợi suất cổ tức | fact_cash_flow.dividends_paid, fact_daily_market.close → mart_master_analysis.dividend_yield |
| **Growth YoY** | (current − previous) / previous * 100 | Tăng trưởng YoY | Window LAG trong SQL → mart_master_analysis.revenue_growth_yoy |
| **CFO/Revenue** | CFO / Revenue | Chất lượng dòng tiền | fact_cash_flow.cfo, fact_income_statement.revenue → mart_master_analysis.cfo_to_revenue |
| **Accrual Ratio** | (NI − CFO) / total_assets | Chất lượng lợi nhuận | net_income, cfo, total_assets → mart_master_analysis.accrual_ratio |
| **Current Ratio** | total_current_assets / total_current_liabilities | Khả năng thanh toán ngắn hạn | fact_balance_sheet → mart_master_analysis.current_ratio |
| **D/E Ratio** | (short_debt + long_debt) / equity | Rủi ro nợ | fact_balance_sheet → mart_master_analysis.debt_to_equity |
| **Cash Conversion Cycle** | DaysInv + DaysSales − DaysPay | Chu kỳ vốn lưu động | tính từ vòng quay → mart_master_analysis.cash_conversion_cycle |
| **Altman Z-Score** | 1.2 (WC / TA) + 1.4 (RE / TA) + 3.3 (EBIT / TA) + 0.6 (MV / BV) + 1.0 (Sales / TA) | Dự báo phá sản | tổng hợp Balance Sheet + Income + Market → mart_master_analysis.altman_z_score |
| **RSI** | 100 − (100 / (1 + RS)) | Chỉ báo kỹ thuật 14 ngày | tính on-the-fly từ fact_daily_market |
| **MACD** | EMA12 − EMA26 | Chỉ báo động lượng | tính on-the-fly từ fact_daily_market.close |
| **YTM** | solve(NPV = 0) | Lợi suất đáo hạn trái phiếu | code_execution (scipy.optimize) |
| **IRR** | internal rate s.t NPV=0 | Suất hoàn vốn nội bộ | numpy.irr |
| **NPV** | Σ(CFt / (1 + r) ^ t) − InitialCost | Giá trị hiện tại ròng | numpy.npv |
| **Kd** | Interest / TotalDebt * (1−Tax) | Chi phí nợ sau thuế | từ Interest Expense & Debt |
| **Ke** | Rf + Beta (Rm − Rf) | Chi phí vốn chủ (CAPM) | Rf (macro), Rm market, Beta → mart_master |
| **WACC** | (E / V) Ke + (D / V) Kd (1 − T) | Chi phí vốn bình quân | dùng E, D từ balance sheet |
| **Interest Coverage** | EBIT / Interest Expense | Khả năng trả lãi | mart_master_analysis.interest_coverage |
| **Short-Term Debt Ratio** | ShortTermDebt / TotalDebt | Tỷ trọng nợ ngắn hạn | fact_balance_sheet |
| **Long-Term Debt Ratio** | LongTermDebt / TotalDebt | Tỷ trọng nợ dài hạn | fact_balance_sheet |
| **Beneish M-Score** | đa biến (DSRI, GMI, AQI…) | Phát hiện gian lận BCTC | tính on-the-fly / code_execution |


Ngoài ra, bond_data nên gồm trường market_rate (lãi suất thị trường hiện hành) để tính giá trị hiện tại của trái phiếu:
```bash
Present Value = Σ (CashFlow_t / (1 + market_rate) ^ t).
```
Cần bổ sung trong OLAP logic của use case Bond Analysis.

===========================

## A. Người Phân Tích / Chuyên Gia Tài Chính (Analyst)
### Use Case A1 — Tóm tắt tình hình tài chính doanh nghiệp

- **Mục tiêu nghiệp vụ:** Cung cấp dashboard tổng quan về báo cáo KQKD (IS), bảng cân đối kế toán (BS), dòng tiền (CF) và thị trường (market) của doanh nghiệp. Bao gồm các chỉ số tài chính chính và chú thích ngắn gọn. Edge cases: Nếu kỳ dữ liệu yêu cầu không có, chuyển sang kỳ gần nhất; Nếu nhập nhiều ticker, hiển thị theo từng ticker hoặc tổng hợp.

- **Input:** Mã cổ phiếu (dim_company.symbol, ví dụ ‘VCB’), Kỳ báo cáo (dim_period: year=2024, quarter=4, period_type có thể là 'Q', 'YTD', 'TTM', 'Y').

- **Output:** Bảng báo cáo (Markdown/JSON) gồm:

    - **KQKD:** Doanh thu (fact_income_statement.revenue), Lợi nhuận ròng (net_income), EPS.

    - **Tỷ số chính:** ROE_TTM, ROA_TTM, CFO/Revenue, Biên lợi nhuận gộp (Gross Margin), Market Cap (đơn vị tỷ VNĐ).

    - **Dòng tiền:** Giải thích dòng tiền hoạt động (CFO), đầu tư (CFI), tài chính (CFF) (fact_cash_flow).

    - **Nợ:** Tỷ lệ nợ ngắn hạn/dài hạn (short/long debt ratio).

    - **Narrative:** Mô tả ngắn: ví dụ "Doanh thu tăng X% so với cùng kỳ, ROE cao, tuy nhiên tỷ lệ nợ ngắn hạn lớn có thể rủi ro".

- **OLAP/Data Source:** Sử dụng bảng mart_master_analysis để lấy KPI tính sẵn; bảng fact_cash_flow cho chi tiết dòng tiền; bảng fact_balance_sheet để lấy cấu trúc nợ; fact_daily_market để cập nhật giá và Market Cap; dim_period để lọc kỳ.

- **Logic/Tính Toán:** Sử dụng chỉ số pre-calculated trong mart_master_analysis. Tính TTM bằng cách cộng 4 quý gần nhất (SUM over window). Tỷ lệ nợ: short_term_debt / (short_term_debt + long_term_debt). Nếu số liệu null, gán 0 hoặc chú thích “không có dữ liệu”. Nếu cần, thực hiện code_execution để tính tổng hoặc xử lý tùy biến.

- **Ví dụ SQL:**
```sql
SELECT m.revenue, m.net_income, m.eps, m.roe_ttm, m.roa_ttm, m.cfo_to_revenue, m.gross_margin, m.market_cap_b,
    c.cfo, c.cfi, c.cff,
    b.short_term_debt / (b.short_term_debt + b.long_term_debt) AS short_debt_ratio
FROM mart_master_analysis m
LEFT JOIN fact_cash_flow c ON m.symbol=c.symbol AND m.report_date=c.report_date
LEFT JOIN fact_balance_sheet b ON m.symbol=b.symbol AND m.report_date=b.report_date
WHERE m.symbol='VCB' AND m.year=2024 AND m.quarter=4;
-- Tính TTM: SUM qua 4 quý gần nhất sử dụng WINDOW function nếu cần.
```
- **RAG Prompt:** “Tóm tắt tình hình tài chính VCB Q4-2024, bao gồm các KPIs chính, phân tích dòng tiền (CFO/CFI/CFF) và cấu trúc nợ.”

### Use Case A2 — So sánh chỉ số tài chính giữa các kỳ

- **Mục tiêu nghiệp vụ:** Theo dõi xu hướng YoY/QoQ của các chỉ số (doanh thu, biên lợi nhuận, v.v.) và phát hiện bất thường (tăng/giảm quá 20%). Edge cases: Thiếu kỳ trước thì ghi chú; Hình thức so sánh tùy chọn (YoY hoặc QoQ).

- **Input:** Mã cổ phiếu, Danh sách các kỳ cần so sánh (mảng dim_period, ví dụ [(2023, Q3), (2023, Q4), (2024, Q1)]), Loại so sánh ('YoY' hoặc 'QoQ').

- **Output:** Bảng theo từng kỳ gồm: kỳ (nhãn Year-Quarter), Doanh thu, Tốc độ tăng trưởng (%), Biên lợi nhuận. Kèm biểu đồ trend JSON. Thông báo cảnh báo nếu có thay đổi đột biến (>20%).

- **OLAP/Data Source:** Từ mart_master_analysis (có growth YoY tính sẵn) kết hợp dim_period để lọc.

- **Logic/Tính Toán:** Tính tăng trưởng = (Giá trị kỳ này - Giá trị kỳ trước) / Giá trị kỳ trước * 100. Phát hiện bất thường: if |growth| > 20%. Nếu kỳ trước không có, đánh dấu 0% hoặc ghi chú.

- **Ví dụ SQL:**
```sql
SELECT CONCAT(year,'-Q',quarter) AS period, revenue, revenue_growth_yoy, net_margin
FROM mart_master_analysis
WHERE symbol='HPG' AND (year, quarter) IN ((2023,3),(2023,4),(2024,1),(2024,2))
ORDER BY year, quarter;
-- Dễ dàng chuyển sang QoQ: dùng LAG(revenue) OVER theo thứ tự ngày.
```
- **RAG Prompt:** “So sánh doanh thu và biên lợi nhuận của HPG qua 4 quý gần nhất, phát hiện bất thường nếu có.”

### Use Case A3 — Phân tích hiệu quả hoạt động (DuPont)

- **Mục tiêu nghiệp vụ:** Giải thích các chỉ số ROE, ROA, ROIC bằng cách phân tích nhân tố (DuPont). Edge cases: Nếu chỉ số âm (lỗ), cảnh báo hiệu quả kém.

- **Input:** Mã cổ phiếu, Kỳ (dim_period: year=2024, quarter=0 cho cả năm).

- **Output:** Các chỉ số: ROE_TTM, ROA_TTM, ROIC; Phân tích chi tiết: Biên lợi nhuận (net_margin), Vòng quay tài sản (asset_turnover), Hệ số đòn bẩy (equity_multiplier). Narrative giải thích: "ROE tăng nhờ biên lợi nhuận cải thiện; tài sản sử dụng hiệu quả kém dẫn đến ROA thấp."

- **OLAP/Data Source:** mart_master_analysis (có các chỉ số pre-calc: dupont_roe, net_margin, asset_turnover, equity_multiplier).

- **Logic/Tính Toán:** DuPont: ROE = net_margin * asset_turnover * equity_multiplier. Đối chiếu số liệu hiện tại với kỳ trước để xác định nguyên nhân tăng/giảm.

- **Ví dụ SQL:**
```sql
SELECT roe_ttm, roa_ttm, roic, net_margin, asset_turnover, equity_multiplier
FROM mart_master_analysis
WHERE symbol='VCB' AND year=2024 AND quarter=0;
```
- **RAG Prompt:** “Phân tích ROE, ROA, ROIC của VCB năm 2024 bằng phương pháp DuPont: chia nhỏ thành biên lợi nhuận, vòng quay tài sản, đòn bẩy vốn.”

### Use Case A4 — Định giá doanh nghiệp (Valuation)

- **Mục tiêu nghiệp vụ:** So sánh các chỉ số định giá (PE, PB, EV/EBITDA) của công ty với mức trung vị ngành; Tính WACC để dùng trong DCF. Edge cases: Nếu không có dữ liệu ngành, dùng mức trung bình thị trường.

- **Input:** Mã cổ phiếu, Ngành (dim_company.sector), Kỳ (dim_period).

- **Output:** Bảng gồm PE_TTM, PB, EV/EBITDA, WACC; Sai khác so với trung vị/ngành (%). Narrative: ví dụ "HPG đang được định giá thấp hơn trung vị ngành thép, khả năng undervalued."

- **OLAP/Data Source:** mart_master_analysis (chỉ số của công ty); mart_master_analysis + dim_company để tính trung vị/độ lệch ngành. Bảng fact_balance_sheet để lấy vốn hóa và nợ.

- **Logic/Tính Toán:** Tính giá trị trung vị/độ lệch: (giá công ty - median) / median * 100. Tính WACC on-the-fly:

    - Lấy Ke = Rf + beta*(Rm - Rf) (Rf, Rm từ forecast_table hoặc fact_macro_timeseries, beta từ mart).

    - Lấy Kd = (interest_expense / total_debt) * (1 - tax_rate).

    - WACC = (E/V)*Ke + (D/V)Kd(1-tax_rate).

- **Ví dụ SQL:**
```sql
WITH sector_med AS (
SELECT MEDIAN(pe_ttm) AS med_pe
FROM mart_master_analysis m
JOIN dim_company d ON m.symbol=d.symbol
WHERE d.sector='Steel' AND m.year=2024
)
SELECT m.pe_ttm, m.pb, m.ev_ebitda,
    ((m.pe_ttm - s.med_pe)/s.med_pe)*100 AS dev_pe,
    ((b.total_equity / v) * ke) + ((debt / v) * kd * (1-0.2)) AS wacc
FROM mart_master_analysis m
JOIN fact_balance_sheet b ON m.symbol=b.symbol AND m.report_date=b.report_date
CROSS JOIN sector_med s
WHERE m.symbol='HPG' AND m.year=2024;
```
- **RAG Prompt:** “Định giá HPG so với trung vị ngành thép: trình bày PE, PB, EV/EBITDA và WACC (từ Ke/Kd). Đánh giá undervalued hay overvalued.”

### Use Case A5 — Benchmark so với ngành/nhóm (Peer Comparison)

- **Mục tiêu nghiệp vụ:** So sánh các KPI chính của công ty với giá trị trung bình hoặc trung vị của ngành, xếp hạng trong ngành. Edge cases: Chọn KPI linh hoạt.

- **Input:** Mã cổ phiếu, Ngành, Kỳ.

- **Output:** Bảng các KPI: Giá trị công ty, Trung bình/Trung vị ngành, Sai khác (%). Thứ hạng trong ngành (rank). Narrative: ví dụ "VCB có ROE cao hơn 95% ngân hàng, dẫn đầu nhóm".

- **OLAP/Data Source:** mart_master_analysis nhóm theo sector (với dim_company), để lấy trung bình hoặc trung vị.

- **Logic/Tính Toán:** Tính trung bình/med cho từng KPI theo sector. Tính sai khác: (company - avg)/avg * 100. Tính rank: ROW_NUMBER() OVER (PARTITION BY sector ORDER BY KPI DESC).

- **Ví dụ SQL:**
```sql
WITH sector_stats AS (
SELECT AVG(pe_ttm) AS avg_pe, AVG(roe_ttm) AS avg_roe
FROM mart_master_analysis m
JOIN dim_company d ON m.symbol=d.symbol
WHERE d.sector='Banking' AND m.year=2024
)
SELECT m.pe_ttm, s.avg_pe, ((m.pe_ttm - s.avg_pe)/s.avg_pe)*100 AS dev_pe
FROM mart_master_analysis m
CROSS JOIN sector_stats s
WHERE m.symbol='VCB';
```
Xếp hạng PE trong ngành:
```sql
SELECT symbol, pe_ttm, 
    ROW_NUMBER() OVER (PARTITION BY sector ORDER BY pe_ttm) AS rank
FROM mart_master_analysis m
JOIN dim_company d ON m.symbol=d.symbol
WHERE d.sector='Banking' AND m.year=2024;
```
- **RAG Prompt:** “So sánh ROE, PE của VCB với các ngân hàng cùng ngành. Trình bày giá trị của VCB, trung bình ngành và sai khác, kèm đánh giá vị trí.”

### Use Case A6 – Đánh giá lợi nhuận hoạt động

- **Mục tiêu:** Tính toán lợi nhuận hoạt động (EBIT) và biên lợi nhuận hoạt động, phản ánh chi phí khấu hao.

- **Đầu vào:** Doanh thu, giá vốn hàng bán (COGS), chi phí hoạt động khác (SG&A), chi phí khấu hao tài sản cố định và tài sản vô hình.

- **Đầu ra:** Lợi nhuận thuần từ hoạt động (EBIT) và tỷ suất lợi nhuận hoạt động (EBIT / doanh thu).

- **Logic tính toán:**

    - EBIT được tính bằng cách trừ COGS, SG&A và khấu hao từ doanh thu. Nói cách khác, EBIT = EBITDA – D&A. Nếu có EBITDA (tính trước khấu hao), thì EBIT = EBITDA – (Depreciation + Amortization).

    - **Ví dụ:** Doanh thu 500, COGS 200, SG&A 50, khấu hao 30, thì EBIT = 500 – 200 – 50 – 30 = 220. Biên lợi nhuận hoạt động = 220/500.

    - **Cần lưu ý:** Khi theo IFRS 16, chi phí thuê dài hạn không còn tính vào SG&A mà thành khấu hao + lãi vay, dẫn tới EBIT tăng (vì ta không trừ chi phí thuê trực tiếp).

- **Mẫu SQL:**
```sql
SELECT 
revenue - COGS - SGandA - depreciation AS operating_profit,
(revenue - COGS - SGandA - depreciation) / revenue AS operating_margin
FROM profit_loss
WHERE company_id = :company_id AND period = :period;
```
- **Prompt (RAG):** Ví dụ: “Cho báo cáo tài chính: Doanh thu 1.000 tỷ, COGS 600 tỷ, SG&A 100 tỷ, khấu hao tài sản 50 tỷ. Hỏi lợi nhuận hoạt động và biên lợi nhuận hoạt động.” Hệ thống sẽ tính EBIT = 250 (bằng 1.000−600−100−50) và biên = 25%.

### Use Case A7 — Phân tích ngang (Horizontal Analysis)

- **Mục tiêu nghiệp vụ:** Tính % thay đổi của các mục (doanh thu, chi phí…) giữa các kỳ để nhận diện xu hướng và bất thường (>20%). Edge cases: Kỳ gốc (base) nếu thiếu thì chọn kỳ đầu hoặc ghi chú.

- **Input:** Mã cổ phiếu, Danh sách nhiều kỳ liên tiếp (dim_period).

- **Output:** Bảng: Kỳ, Doanh thu, % thay đổi (YoY hoặc so kỳ trước), Chi phí, % thay đổi. Thông báo mục bất thường. Narrative khái quát.

- **OLAP/Data Source:** mart_master_analysis (có doanh thu, các khoản mục).

- **Logic/Tính Toán:** Tính % change = (current - previous) / previous *100 dùng LAG trong SQL. Ghi dấu hiệu tăng/giảm đột biến nếu |%change| > 20%.

- **Ví dụ SQL:**
```sql
SELECT year, quarter, revenue,
    ((revenue - LAG(revenue) OVER (ORDER BY year,quarter)) / LAG(revenue) OVER (ORDER BY year,quarter)) *100 AS revenue_change,
    expenses, ((expenses - LAG(expenses) OVER (ORDER BY year,quarter)) / LAG(expenses) OVER (ORDER BY year,quarter)) *100 AS expenses_change
FROM mart_master_analysis
WHERE symbol='VCB' AND year BETWEEN 2023 AND 2024;
```
- **RAG Prompt:** “Horizontal analysis KQKD VCB 2023-2024: thể hiện thay đổi phần trăm của doanh thu và chi phí qua các kỳ.”

### Use Case A8 — Phân tích cơ cấu (Vertical/Common-Size Analysis)

- **Mục tiêu nghiệp vụ:** Xác định tỷ trọng từng khoản mục trong cùng một báo cáo. Edge cases: Tổng = 0 thì null.

- **Input:** Mã cổ phiếu, Kỳ, Loại báo cáo ('IS' hoặc 'BS').

- **Output:** Bảng: Mục (ví dụ Revenue, COGS, etc), Giá trị tuyệt đối, Tỷ lệ % trên tổng (ví dụ COGS/Revenue*100).

- **OLAP/Data Source:** fact_income_statement hoặc fact_balance_sheet tùy yêu cầu.

- **Logic/Tính Toán:** Tính % = item / total * 100 (ví dụ: COGS / Revenue * 100). Với bảng BS, thường so sánh với tổng tài sản.

- **Ví dụ SQL:**
```sql
SELECT 'Doanh thu' AS item, revenue AS value, 100 AS pct
UNION ALL
SELECT 'COGS', cogs, cogs/revenue*100
FROM fact_income_statement
WHERE symbol='HPG' AND year=2024 AND quarter=4;
```
- **RAG Prompt:** “Phân tích cơ cấu Bảng Cân đối kế toán HPG Q4-2024: liệt kê các tài sản, nợ và vốn chủ có giá trị và tỷ trọng.”

### Use Case A9 — Phân tích khả năng thanh toán & rủi ro nợ

- **Mục tiêu nghiệp vụ:** Đánh giá các tỷ số thanh toán (Current Ratio) và rủi ro nợ (D/E, Interest Coverage, Kd). Edge cases: Kiểm tra overdraft từ ghi chú.

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** Các tỷ số: D/E, Current Ratio, Interest Coverage, Kd (chi phí nợ sau thuế), short-term debt ratio, long-term debt ratio. Narrative đánh giá rủi ro (ví dụ: D/E > 2 là cao).

- **OLAP/Data Source:** mart_master_analysis (có các tỷ lệ: debt_to_equity, current_ratio, interest_coverage), fact_balance_sheet (nợ ngắn/dài hạn), fact_income_statement (interest_expense).

- **Logic/Tính Toán:** Kd = (interest_expense / (short_term_debt + long_term_debt)) * (1 - tax_rate). Các tỷ lệ khác có sẵn trong mart. Nếu mất số liệu, dùng mark 0.

- **Ví dụ SQL:**
```sql
SELECT m.debt_to_equity, m.current_ratio, m.interest_coverage,
    (i.interest_expense / (b.short_term_debt + b.long_term_debt)) * (1 - 0.2) AS kd
FROM mart_master_analysis m
JOIN fact_income_statement i ON m.symbol=i.symbol AND m.report_date=i.report_date
JOIN fact_balance_sheet b ON m.symbol=b.symbol AND m.report_date=b.report_date
WHERE m.symbol='VCB' AND m.year=2024;
```
- **RAG Prompt:** “Đánh giá rủi ro thanh toán nợ VCB 2024: các tỷ số D/E, Current Ratio, Interest Coverage, bao gồm tính chi phí nợ (Kd) và phân tích quá mức tín dụng (overdraft).”

### Use Case A10 — Xác định Break-Even / Điểm hòa vốn

- **Mục tiêu nghiệp vụ:** Tính điểm hòa vốn trên doanh thu và phân tích nhạy cảm. Edge cases: Ước tính chi phí cố định nếu thiếu.

- **Input:** Mã cổ phiếu, Kỳ; Chi phí cố định và biến đổi (nếu không có có thể lấy xấp xỉ từ SG&A và COGS).

- **Output:** Điểm hòa vốn (doanh thu). Phân tích nhạy cảm: ví dụ doanh thu cần thiết tăng X% nếu chi phí tăng 10%. Narrative.

- **OLAP/Data Source:** fact_income_statement (SG&A ~ fixed costs, COGS ~ variable costs).

- **Logic/Tính Toán:** Break-even = FixedCost / (1 - VariableCost/Revenue). Tính sensitivity bằng code_execution: tăng fixed hoặc variable +10% ảnh hưởng ra sao.

- **Ví dụ SQL:**
```sql
SELECT (sgna + interest_expense) / (1 - cogs / revenue) AS break_even
FROM fact_income_statement
WHERE symbol='HPG' AND year=2024;
```
- **RAG Prompt:** “Điểm hòa vốn HPG Q4-2024 (với giả định SG&A+Lãi vay là chi phí cố định, COGS là biến đổi), và ảnh hưởng nếu chi phí tăng 10%.”

### Use Case A11 — Tạo báo cáo MD&A-style tự động

- **Mục tiêu nghiệp vụ:** Viết narration giải thích biến động KPI chính và liên kết với yếu tố vĩ mô. Edge cases: Giải thích nhiều KPI, tránh dài.

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** Đoạn văn bản MD&A (Management Discussion & Analysis) nêu lý do biến động doanh thu, lợi nhuận, mảng chính, gắn với chỉ số vĩ mô (lạm phát, lãi suất…).

- **OLAP/Data Source:** mart_master_analysis (growth, margin), fact_macro_timeseries (CPI, Lãi suất, GDP…).

- **Logic/Tính Toán:** Lấy các chỉ số tăng trưởng từ mart. Tính hệ số tương quan (on-the-fly regression) giữa KPI với chỉ số vĩ mô.

- **Ví dụ SQL:**
```sql
SELECT m.revenue_growth_yoy, m.net_margin,
    (SELECT value FROM fact_macro_timeseries 
    WHERE indicator_code='VN_CPI_YOY' AND YEAR(date)=m.year) AS cpi
FROM mart_master_analysis m
WHERE m.symbol='VCB' AND m.year=2024;
```
- **RAG Prompt:** “Viết đoạn MD&A cho VCB năm 2024: giải thích biến động doanh thu, lợi nhuận dựa trên các KPI chính và tác động của CPI, lãi suất.”

### Use Case A12 — Phân tích DuPont (mở rộng ROE)

- **Mục tiêu nghiệp vụ:** Xác định thành phần cấu thành ROE và biến động qua các năm. Edge cases: Thể hiện cả yếu tố nợ (đòn bẩy).

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** ROE, net_margin, asset_turnover, equity_multiplier; sự thay đổi YoY của từng thành phần; nhận xét.

- **OLAP/Data Source:** mart_master_analysis (dupont_roe, net_margin, asset_turnover, equity_multiplier).

- **Logic/Tính Toán:** ROE = margin * turnover * multiplier. Tính Δ = hiện tại - trước.

- **Ví dụ SQL:**
```sql
SELECT dupont_roe AS roe, net_margin, asset_turnover, equity_multiplier
FROM mart_master_analysis
WHERE symbol='VCB' AND year=2024;
```
- **RAG Prompt:** “Phân tích chi tiết DuPont ROE cho VCB 2024: nêu ROE cùng các thành phần, đồng thời so sánh thay đổi YoY.”

### Use Case A13 — Altman Z-Score (Đánh giá rủi ro phá sản)

- **Mục tiêu nghiệp vụ:** Tính Z-Score và phân loại rủi ro. Edge cases: Z < 1.8 (rủi ro cao), >3 (an toàn).

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** Z-Score (con số) và mức độ (Safe/Gray/Distress); Phân tích các thành phần cấu tạo (WC/TA, RE/TA, etc.).

- **OLAP/Data Source:** mart_master_analysis (altman_z_score).

- **Logic/Tính Toán:** Lấy giá trị từ mart; Xác định cấp độ: IF(z>3) Safe; IF(z<1.8) Distress; else Gray.

- **Ví dụ SQL:**
```sql
SELECT altman_z_score,
    CASE 
        WHEN altman_z_score > 3 THEN 'Safe'
        WHEN altman_z_score < 1.8 THEN 'Distress'
        ELSE 'Gray'
    END AS risk_level
FROM mart_master_analysis
WHERE symbol='HPG' AND year=2024;
```
- **RAG Prompt:** “Tính Altman Z-Score cho HPG năm 2024 và đánh giá rủi ro phá sản.”

### Use Case A14 – Định giá trái phiếu

- **Mục tiêu:** Tính toán giá trị hiện tại (PV) của trái phiếu (coupon bond) dựa trên lãi suất thị trường.

- **Đầu vào:** Dữ liệu trái phiếu gồm mệnh giá (face_value), lãi suất coupon (coupon_rate), tần suất trả lãi, thời gian đáo hạn (số năm), và lãi suất thị trường (market_rate) mới bổ sung.

- **Đầu ra:** Giá trị hiện tại của trái phiếu (present_value).

- **Logic tính toán:** Áp dụng công thức chiết khấu dòng tiền: present value bằng tổng của giá trị hiện tại của các khoản thanh toán lãi hàng kỳ và mệnh giá khi đáo hạn. Cụ thể:
```bash
PV = C × (1 – (1 + r)^(-n))/ r + F / (1 + r)^n
```

trong đó C là khoản thanh toán coupon định kỳ, r = market_rate là lãi suất chiết khấu (yield), n là số kỳ, F là mệnh giá. Ví dụ, với trả lãi hàng năm, C = coupon_rate × face_value. Công thức này phản ánh rõ việc chiết khấu dòng coupon và mệnh giá về giá trị hiện tại.

- **Mẫu SQL:** Ví dụ giả định nếu tính PV cho các năm một cách tuần tự:
```sql
SELECT bond_id,
SUM(coupon_payment / POWER(1 + market_rate, seq)) 
+ face_value / POWER(1 + market_rate, total_periods) AS present_value
FROM bond_cashflows
WHERE bond_id = :bond_id
GROUP BY bond_id;
```
Hoặc trường hợp đơn giản:
```sql
SELECT face_value,
    coupon_rate,
    maturity_years,
    market_rate,
    (coupon_rate * face_value * (1 - POWER(1 + market_rate, -maturity_years)) / market_rate)
    + (face_value / POWER(1 + market_rate, maturity_years)) AS present_value
FROM bond_data
WHERE bond_id = :bond_id;
```
- **Prompt (RAG):** Ví dụ: “Cho trái phiếu A có mệnh giá 1.000, coupon 5%/năm, đáo hạn sau 5 năm và lãi suất thị trường 4%. Tính giá trị hiện tại của trái phiếu.” Hệ thống sẽ đọc giá trị của market_rate và sử dụng công thức chiết khấu để trả về PV.

### Use Case A15 — Tính WACC từ Kd/Ke và phân tích nhạy cảm

- **Mục tiêu nghiệp vụ:** Tính chi phí vốn bình quân (WACC) của doanh nghiệp, đánh giá sự nhạy cảm của WACC với biến động tham số (e.g. beta).

- **Input:** Mã cổ phiếu, Kỳ (có thể sử dụng forecast_year từ forecast_table).

- **Output:** WACC, Ke (chi phí vốn chủ), Kd (chi phí nợ); Kết quả phân tích nhạy cảm: WACC khi beta tăng/giảm 0.1; Narrative.

- **OLAP/Data Source:** mart_master_analysis (beta), fact_balance_sheet (E, D để tính tỷ trọng), forecast_table (Rf, Rm; nếu không có, dùng fact_macro SBV_RATE + 2%), thuế suất (từ extra items hoặc mặc định 20%).

- **Logic/Tính Toán:**

    - Lấy Rf và Rm từ forecast_table theo forecast_year (fallback sang lãi suất thực tế nếu thiếu).

    - Tính Ke = Rf + beta * (Rm - Rf).

    - Tính Kd = (interest_expense / total_debt) * (1 - tax_rate).

    - Tính WACC = (E/V)*Ke + (D/V)Kd(1-tax_rate).

    - Phân tích nhạy cảm: lặp tính WACC với beta ±0.1 (code_execution).

- **Ví dụ SQL:**
```sql
SELECT (b.total_equity / v) * ke + (b.short_term_debt + b.long_term_debt) / v * kd * (1-0.2) AS wacc
FROM (
SELECT total_equity + short_term_debt + long_term_debt AS v, total_equity, short_term_debt, long_term_debt
FROM fact_balance_sheet
WHERE symbol='HPG' AND year=2024
) b
JOIN forecast_table f ON b.symbol=f.symbol
WHERE f.year=2024;
```
- **RAG Prompt:** “Tính WACC cho HPG (theo dữ liệu forecast): cho biết Ke, Kd và WACC; phân tích nhạy cảm khi beta thay đổi ±0.1.”

### Use Case A16 — Phân tích phát hiện gian lận (Beneish M-Score)

- **Mục tiêu nghiệp vụ:** Xác định nguy cơ gian lận bằng chỉ số M-Score. Edge cases: M-Score > -2.22 nghi ngờ.

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** M-Score, các thành phần DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA; Cảnh báo nếu M-Score vượt ngưỡng. Narrative.

- **OLAP/Data Source:** Các chỉ số tăng trưởng, biên, tài sản trong mart_master_analysis.

- **Logic/Tính Toán:** Sử dụng công thức chuẩn của Beneish (dùng code_execution):
```bash
M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
```
Mỗi thành phần tính theo tỷ lệ các chỉ tiêu kế toán giữa 2 kỳ.

- **Ví dụ SQL:**
```sql
SELECT receivables / revenue AS DSRI_cur, prev_receivables/prev_revenue AS DSRI_prev,
    net_income - cfo AS TAC,
    -- lấy các tỷ lệ cần thiết để tính thành phần
FROM mart_master_analysis
WHERE symbol='VCB';
```
Sau đó tính M-Score bằng Python.

- **RAG Prompt:** “Tính M-Score cho VCB để xác định rủi ro gian lận. Liệt kê các thành phần và cảnh báo nếu cần.”

===========================

## B. Nhà Đầu Tư / Trader
### Use Case B1 — Phân tích biến động giá & thị trường

- **Mục tiêu nghiệp vụ:** Đánh giá biến động giá ngắn hạn (volatility), khối lượng giao dịch trung bình và Beta so với chỉ số thị trường. Edge cases: Thời gian linh hoạt (mặc định 30 ngày).

- **Input:** Mã cổ phiếu, Khung thời gian (ví dụ '30d').

- **Output:** Độ biến động giá (stddev 30d), khối lượng bình quân, Beta; Phân phối lợi suất JSON; Narrative.

- **OLAP/Data Source:** fact_daily_market (giá, volume, ngày), fact_macro_timeseries (VN_INDEX).

- **Logic/Tính Toán:**
```bash
Volatility = STDDEV_POP(close) trong khung thời gian.

Avg_volume = AVG(volume).

Tính lợi suất hàng ngày, sau đó Beta = Cov(stock_ret, index_ret) / Var(index_ret). Dùng index VN_INDEX từ fact_macro_timeseries.
```
- **Ví dụ SQL:**
```sql
SELECT STDDEV_POP(close) AS vol_30d, AVG(volume) AS avg_vol
FROM fact_daily_market
WHERE symbol='HPG' AND date >= today() - 30;
```
Tính Beta (pseudocode):
```sql
WITH ret AS (
SELECT date,
        (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) AS stock_ret
FROM fact_daily_market
WHERE symbol='HPG' AND date >= today() - 30
)
SELECT covar_pop(stock_ret, index_ret)/var_pop(index_ret) AS beta
FROM ret JOIN fact_macro_timeseries m ON ret.date = m.date
WHERE m.indicator_code='VN_INDEX';
```
- **RAG Prompt:** “Tính biến động giá (30d), khối lượng bình quân và Beta (so với VN-Index) của HPG.”

### Use Case B2 — Dự báo & cảnh báo tín hiệu

- **Mục tiêu nghiệp vụ:** Cảnh báo nếu KPI hoặc giá có biến động theo quy tắc tùy chỉnh. Edge cases: Cho phép người dùng nhập quy tắc (ví dụ: EPS giảm >10%, Kd tăng >5%).

- **Input:** Mã cổ phiếu, Bộ quy tắc (ví dụ {'eps_drop':-10%, 'kd_rise':5%}).

- **Output:** Danh sách cảnh báo: KPI nào vi phạm, mức độ, kỳ xảy ra.

- **OLAP/Data Source:** mart_master_analysis (EPS, Kd, v.v.), bond_data (nếu có cảnh báo trái phiếu YTM).

- **Logic/Tính Toán:** Áp dụng điều kiện lọc SQL hoặc code: ví dụ EPS YoY < -10% THEN cảnh báo “EPS giảm mạnh”. Kd tăng trên 5% so với kỳ trước.

- **Ví dụ SQL:**
```sql
SELECT year, quarter
FROM mart_master_analysis
WHERE symbol='VCB' AND eps_growth_yoy < -10;
```
- **RAG Prompt:** “Kiểm tra và cảnh báo nếu EPS giảm liên tiếp 2 quý hoặc Kd tăng trên 5%.”

### Use Case B3 — So sánh cổ phiếu với nhóm peer

- **Mục tiêu:** Xếp hạng các cổ phiếu trong cùng ngành theo KPI (ví dụ PE từ thấp đến cao). Edge cases: Lấy top N (ví dụ 10).

- **Input:** Ngành (dim_company.sector), KPI (ví dụ 'pe_ttm'), Top_n.

- **Output:** Bảng xếp hạng: Mã, Giá trị KPI. Narrative ví dụ “Trong ngành thép, HPG có PE đứng thứ 3 trên 10.”

- **OLAP/Data Source:** mart_master_analysis joined dim_company để lấy sector.

- **Logic/Tính Toán:** ORDER BY KPI (ASC cho PE, DESC cho growth) LIMIT N.

- **Ví dụ SQL:**
```sql
SELECT m.symbol, m.pe_ttm
FROM mart_master_analysis m
JOIN dim_company d ON m.symbol=d.symbol
WHERE d.sector='Steel' AND m.year=2024
ORDER BY m.pe_ttm ASC
LIMIT 10;
```
- **RAG Prompt:** “Xếp hạng 10 cổ phiếu hàng đầu trong ngành thép theo PE thấp nhất.”

### Use Case B4 — Phân tích mùa vụ & TTM chính xác

- **Mục tiêu nghiệp vụ:** Tính toán KPI theo chuỗi 12 tháng (TTM) để loại trừ ảnh hưởng mùa vụ; so sánh TTM với giá trị annual. Edge cases: Nếu thiếu quý, ước tính theo tỷ lệ.

- **Input:** Mã cổ phiếu, Thời gian (có thể đến ngày hiện tại hoặc kỳ cuối).

- **Output:** Doanh thu_TTM, EPS_TTM, so sánh với doanh thu cả năm; Visual/ Narrative.

- **OLAP/Data Source:** mart_master_analysis có thể tổng hợp 4 quý gần nhất.

- **Logic/Tính Toán:** SUM(revenue) qua 4 quý gần nhất từ báo cáo. Nếu năm chưa đủ 4 quý, prorate dựa trên trung bình.

- **Ví dụ SQL:**
```sql
SELECT SUM(revenue) AS revenue_ttm
FROM mart_master_analysis
WHERE symbol='VCB' AND report_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR);
```
- **RAG Prompt:** “Tính doanh thu TTM của VCB và so sánh với doanh thu cả năm để loại bỏ ảnh hưởng mùa vụ.”

### Use Case B5 — So sánh hiệu quả cổ phiếu vs benchmark thị trường

- **Mục tiêu nghiệp vụ:** Tính alpha, beta và relative return so với chỉ số (ví dụ VN-Index). Edge cases: Cho phép chỉ định benchmark khác.

- **Input:** Mã cổ phiếu, Thời gian (ví dụ 1 năm).

- **Output:** Beta, Alpha, Tổng lợi suất so với index. Narrative: ví dụ “HPG vượt VN-Index X% sau 1 năm.”

- **OLAP/Data Source:** fact_daily_market (giá cổ phiếu), fact_macro_timeseries (VN_INDEX), Rf (từ SBV_RATE).

- **Logic/Tính Toán:** Tính lợi suất cổ phiếu và chỉ số. Beta = cov/var. Alpha = stock_ret - [Rf + beta*(index_ret - Rf)]. Relative return = (stock_return - index_return).

- **Ví dụ SQL:**
```sql
WITH ret AS (
    SELECT (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) AS stock_ret, date
    FROM fact_daily_market WHERE symbol='HPG' AND date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
), index_ret AS (
    SELECT (value - LAG(value) OVER (ORDER BY date)) / LAG(value) OVER (ORDER BY date) AS index_ret, date
    FROM fact_macro_timeseries WHERE indicator_code='VN_INDEX' AND date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
)
SELECT covariancePop(stock_ret, index_ret)/variancePop(index_ret) AS beta
FROM ret JOIN index_ret ON ret.date=index_ret.date;
```
Tính Alpha trên Python.

- **RAG Prompt:** “So sánh hiệu suất HPG và VN-Index trong 1 năm: tính beta, alpha và chênh lệch lợi suất.”

### Use Case B6 — Alerts dựa trên trigger tài chính

- **Mục tiêu nghiệp vụ:** Đặt ngưỡng cảnh báo cho các chỉ số (ví dụ net_margin, D/E…). Edge cases: Tùy chỉnh lịch sử.

- **Input:** Mã cổ phiếu, Danh sách rule (ví dụ net_margin giảm 10% so quý trước).

- **Output:** Danh sách sự kiện: Period, KPI, Mức độ cảnh báo.

- **OLAP/Data Source:** mart_master_analysis (các KPI liên quan).

- **Logic/Tính Toán:** Áp dụng điều kiện: e.g. net_margin < LAG(net_margin) -10.

- **Ví dụ SQL:**
```sql
SELECT year, quarter, 'Margin drop' AS note
FROM mart_master_analysis
WHERE symbol='VCB' AND net_margin < LAG(net_margin) OVER (ORDER BY year,quarter) - 10;
```
- **RAG Prompt:** “Xác định các sự kiện vượt ngưỡng: ví dụ biên lợi nhuận VCB giảm hơn 10% so với quý trước.”

### Use Case B7 — RSI / MACD Signal

- **Mục tiêu nghiệp vụ:** Tính RSI, MACD và cảnh báo tín hiệu (quá mua/quá bán). Edge cases: Thời gian tham số tùy chỉnh.

- **Input:** Mã cổ phiếu, Số ngày (ví dụ 14).

- **Output:** Giá trị RSI, MACD; Ký hiệu tín hiệu (RSI>70: overbought, RSI<30: oversold; MACD cross).

- **OLAP/Data Source:** fact_daily_market (giá đóng cửa).

- **Logic/Tính Toán:**

    RSI: Tính average gain/loss 14 ngày, RS, rồi RSI = 100 - 100/(1+RS).

    MACD: Tính EMA12, EMA26, MACD=EMA12-EMA26.

Ví dụ SQL & Mã:
```sql
SELECT close
FROM fact_daily_market
WHERE symbol='HPG' ORDER BY date DESC LIMIT 30;
```
Code tính RSI/MACD với Python:
```py
import talib
prices = [...]
rsi = talib.RSI(prices, timeperiod=14)
macd, signal, hist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
```
- **RAG Prompt:** “Tính RSI và MACD của HPG (14 ngày) và xác định tín hiệu giao dịch.”

### Use Case B8 – Đánh giá lợi suất trái phiếu

- **Mục tiêu:** Xác định các chỉ số lợi suất của trái phiếu, gồm Current Yield và Yield to Maturity (YTM), dựa trên dữ liệu giá thị trường và coupon.

- **Đầu vào:** Dữ liệu trái phiếu gồm giá thị trường hiện tại (market_price), mệnh giá (face_value), lãi suất coupon (coupon_rate), thời gian đến hạn (maturity_years), và có thể sử dụng market_rate làm ngưỡng so sánh.

- **Đầu ra:** Các chỉ số lợi suất:

    - **Current Yield** (lãi suất hiện hành) = (lãi coupon hàng năm) / (giá thị trường).

    - **Yield to Maturity (YTM):** lãi suất nội tại giải phương trình chiết khấu của dòng tiền (có thể xấp xỉ bằng công thức sau):
    ```bash
    YTM= [C+(F−P)/t] / [(F+P)/2]
    ```
    trong đó C là thanh toán coupon hàng năm, F là mệnh giá, P là giá hiện tại, t số năm đáo hạn. YTM chính là lãi suất cho phép tổng PV của tất cả coupon và mệnh giá bằng giá thị trường.

- **Logic tính toán:**

    - **Current Yield:** Giản đơn bằng công thức lãi coupon chia cho giá. Ví dụ: (0.05 × 1.000) / 980 = 5.10%.

    - **YTM:** Giải phương trình nội suy (thử và điều chỉnh lãi suất) sao cho PV của các dòng tiền bằng giá thị trường. Công thức xấp xỉ trên giúp tính nhanh tỷ lệ này.

- **Mẫu SQL:** Ví dụ tính current yield:
```sql
SELECT (coupon_rate * face_value) / market_price AS current_yield
FROM bond_data
WHERE bond_id = :bond_id;
```
Tính approximate YTM:
```sql
SELECT ((coupon_rate * face_value) + (face_value - market_price)/maturity_years)
    / ((face_value + market_price)/2) AS approx_ytm
FROM bond_data
WHERE bond_id = :bond_id;
```
- **Prompt (RAG):** Ví dụ: “Trái phiếu B có mệnh giá 1.000, coupon 5%/năm, đáo hạn 10 năm, hiện giao dịch ở giá 950, trong khi lãi suất thị trường là 6%. Tính Current Yield và Yield to Maturity của trái phiếu.” Hệ thống sẽ tính lãi suất hiện hành và ước lượng YTM theo công thức trên

.

===========================

## C. CEO / CFO / Quản Lý
### Use Case C1 — Kiểm soát rủi ro tài chính nội bộ (Anomaly Detection)

- **Mục tiêu nghiệp vụ:** Phát hiện các dị thường tài chính dựa trên ngưỡng tùy chỉnh (e.g. biên lợi nhuận giảm >20%). Edge cases: Cho phép tùy chỉnh tham số.

- **Input:** Ngưỡng cảnh báo (ví dụ: margin_drop>20%).

- **Output:** Danh sách anomalies: symbol, KPI vi phạm, kỳ.

- **OLAP/Data Source:** mart_master_analysis (net_margin, debt_ratios, v.v.).

- **Logic/Tính Toán:** Dùng SQL điều kiện hoặc code để lọc: e.g. net_margin < LAG(net_margin) -20.

- **Ví dụ SQL:**
```sql
SELECT symbol, year, quarter
FROM mart_master_analysis
WHERE net_margin < LAG(net_margin) OVER (ORDER BY year,quarter) - 20;
```
- **RAG Prompt:** “Tìm các trường hợp biên lợi nhuận giảm >20% trong dữ liệu.”

### Use Case C2 — So sánh kế hoạch vs thực tế (Budget vs Actual)

- **Mục tiêu nghiệp vụ:** So sánh dữ liệu thực tế so với kế hoạch (budget). Edge cases: Nếu thiếu budget, báo No Data.

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** Bảng: Actual vs Plan cho doanh thu, lợi nhuận... và % đạt. Narrative.

- **OLAP/Data Source:** mart_master_analysis (actual KPIs) kết hợp budget_table (KPIs kế hoạch).

- **Logic/Tính Toán:** % đạt = actual / plan * 100. Nếu plan=0, chú ý chia 0.

- **Ví dụ SQL:**
```sql
SELECT m.revenue AS actual, bt.revenue AS plan, (m.revenue/bt.revenue)*100 AS pct_attainment
FROM mart_master_analysis m
JOIN budget_table bt ON m.period_id=bt.period_id AND m.symbol=bt.symbol
WHERE m.symbol='VCB' AND m.year=2024;
```
- **RAG Prompt:** “So sánh doanh thu thực tế và kế hoạch VCB Q4-2024, tính % hoàn thành.”

### Use Case C3 — Phân tích tác động vĩ mô (Correlation)

- **Mục tiêu nghiệp vụ:** Xác định tác động của biến vĩ mô đến KPI chính. Edge cases: Chọn nhiều mã CPI/Lãi suất.

- **Input:** Các mã kinh tế vĩ mô (ví dụ 'VN_CPI_YOY', 'SBV_RATE').

- **Output:** Hệ số tương quan giữa mỗi biến và KPI (net_margin, EPS...), biểu đồ trend. Narrative: ví dụ "CPI tăng có tương quan âm với biên lợi nhuận nhóm ngân hàng."

- **OLAP/Data Source:** fact_macro_timeseries (macro data) kết hợp mart_master_analysis (theo năm).

- **Logic/Tính Toán:** Tính CORR (corr coefficient) giữa giá trị macro năm n với KPI năm n.

- **Ví dụ SQL:**
```sql
SELECT corr(m.value, k.net_margin) AS corr_coef
FROM fact_macro_timeseries m
JOIN mart_master_analysis k ON YEAR(m.date)=k.year
WHERE m.indicator_code='VN_CPI_YOY' AND k.symbol IN (
    SELECT symbol FROM dim_company WHERE sector='Banking'
);
```
- **RAG Prompt:** “Ảnh hưởng của lạm phát (VN_CPI_YOY) đến biên lợi nhuận ròng của nhóm ngân hàng.”

### Use Case C4 — Budget vs Actual trend analysis

- **Mục tiêu nghiệp vụ:** Theo dõi xu hướng thực tế so với kế hoạch theo thời gian.

- **Input:** Mã cổ phiếu, Nhiều kỳ (tháng/quý).

- **Output:** Bảng thời gian: Period, Actual, Budget, % đạt. Biểu đồ dòng. Narrative.

- **OLAP/Data Source:** mart_master_analysis + budget_table.

- **Logic/Tính Toán:** Tương tự Use Case C2 cho từng kỳ.

- **Ví dụ SQL:**
```sql
SELECT m.report_date, m.revenue AS actual, bt.revenue AS plan,
        (m.revenue/bt.revenue)*100 AS pct
FROM mart_master_analysis m
JOIN budget_table bt ON m.period_id=bt.period_id
WHERE m.symbol='VCB'
ORDER BY m.report_date;
```
- **RAG Prompt:** “Xu hướng doanh thu thực tế và kế hoạch VCB từng quý.”

### Use Case C5 — Kiểm tra thanh khoản (Liquidity Stress Test)

- **Mục tiêu nghiệp vụ:** Mô phỏng tình huống stress: ví dụ giả định doanh thu giảm, tiền mặt giảm X%. Edge cases: Đa kịch bản (cash giảm 20%, lãi suất +10%).

- **Input:** Mã cổ phiếu, Kịch bản (danh sách hệ số thay đổi).

- **Output:** Giá trị các tỷ số thanh khoản và “thời gian sống sót” (cash/burn_rate) dưới kịch bản.

- **OLAP/Data Source:** mart_master_analysis (current_ratio, quick_ratio) hoặc fact_balance_sheet (cash, burn_rate).

- **Logic/Tính Toán:** Giảm các số đầu vào: ví dụ current_ratio_new = current_ratio * (1 - factor). Tính cash/burn_rate.

- **Ví dụ SQL:**
```sql
SELECT current_ratio * 0.8 AS stressed_current_ratio
FROM mart_master_analysis
WHERE symbol='HPG' AND year=2024;
```
- **RAG Prompt:** “Giả sử tiền mặt HPG giảm 20%: đánh giá lại tỷ lệ thanh khoản và cash burn.”

### Use Case C6 — What-if & Kịch bản chiến lược (Monte Carlo)

- **Mục tiêu nghiệp vụ:** Mô phỏng kịch bản biến số (thay đổi lãi suất, tỷ giá) ảnh hưởng đến EPS, WACC. Edge cases: Dự báo xác suất (Monte Carlo).

- **Input:** Mã cổ phiếu, Danh sách biến (ví dụ lãi suất +1%).

- **Output:** Giá trị dự báo EPS, WACC theo kịch bản; phân phối đầu ra nếu có Monte Carlo. Biểu đồ.

- **OLAP/Data Source:** mart_master_analysis (EPS cơ sở), forecast_table (dữ liệu macro).

- **Logic/Tính Toán:** Công thức dự báo: ví dụ EPS_new = EPS_base * (1 + impact_of_rate). Dùng numpy.random để chạy mô phỏng phân phối.

- **Ví dụ SQL:**
```sql
SELECT eps * 1.1 AS projected_eps
FROM mart_master_analysis
WHERE symbol='VCB' AND year=2024;
```
Monte Carlo với Python:
```py
import numpy as np
base_eps = 5.0
results = []
for _ in range(10000):
    rate = np.random.normal(0.05, 0.01)  # lãi suất ngẫu nhiên
    results.append(base_eps * (1 - rate))
```
- **RAG Prompt:** “What-if: lãi suất tăng 1% ảnh hưởng EPS, WACC của VCB như thế nào?”

### Use Case C7 — Chu kỳ vòng quay vốn lưu động (CCC)

- **Mục tiêu nghiệp vụ:** Đánh giá hiệu quả sử dụng vốn lưu động: DIO, DSO, DPO, CCC.

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** CCC, giá trị DIO, DSO, DPO; Đề xuất tối ưu (ví dụ giảm DIO).

- **OLAP/Data Source:** mart_master_analysis (có cash_conversion_cycle, days_inventory, days_sales, days_payables).

- **Logic/Tính Toán:** CCC = DIO + DSO - DPO. Phân tích nếu CCC cao, đề xuất giảm tồn kho hoặc thu hồi tiền nhanh hơn.

- **Ví dụ SQL:**
```sql
SELECT days_inventory, days_sales, days_payables,
        (days_inventory + days_sales - days_payables) AS ccc
FROM mart_master_analysis
WHERE symbol='HPG' AND year=2024;
```
- **RAG Prompt:** “Tính CCC của HPG 2024 và đề xuất cải thiện hiệu quả vốn lưu động.”

### Use Case C8 – Phân tích chi phí vốn (Cost of Debt, WACC)

- **Mục tiêu:** Tính toán chi phí sử dụng vốn: tỷ suất chi phí nợ (sau thuế), WACC và hiệu quả sử dụng vốn. Xác định vai trò của D&A trong dòng tiền tính WACC.

- **Đầu vào:** Cơ cấu vốn (tổng nợ, tổng vốn chủ sở hữu), chi phí vốn chủ sở hữu (Re), chi phí vay bình quân (Rd), thuế suất TNDN (Tc), và các yếu tố tính dòng tiền như khấu hao, thu nhập ròng.

- **Đầu ra:** Chi phí nợ (pre-tax và after-tax), WACC, các chỉ tiêu so sánh lợi nhuận với WACC.

- **Logic tính toán:**

    - **Chi phí nợ (Rd):** Tính trung bình lãi suất trả nợ hiện tại (hoặc YTM của trái phiếu công ty). Chi phí nợ sau thuế = Rd × (1 − Tc) vì chi phí lãi vay được khấu trừ thuế. Ví dụ nếu Rd = 10% và Tc = 20%, sau thuế = 8%.

    - **WACC:** Công thức WACC = (E/V)×Re + (D/V)×Rd×(1−Tc). Trong đó E/V và D/V lần lượt là tỷ lệ phần trăm vốn chủ sở hữu và nợ trong tổng nguồn vốn. Ví dụ, nợ 30%, vốn chủ 70%, thì WACC = 0.7Re + 0.3Rd×(1−Tc).

    - **D&A và Dòng tiền:** Để tính dòng tiền tự do (FCF) gắn với WACC, khấu hao được cộng trở lại thu nhập trước thuế khi tính OCF vì không dùng tiền mặt. Đồng thời, khấu hao tạo tax shield (giảm thuế). Tuy nhiên, trong công thức WACC, D&A không xuất hiện trực tiếp; chỉ thuế suất chung (bao gồm lợi ích từ lãi vay và khấu hao) ảnh hưởng đến chi phí vốn thuần.

- **Mẫu SQL:**
```sql
SELECT 
debt_value, equity_value, cost_of_equity, avg_cost_of_debt, tax_rate,
(equity_value/(debt_value+equity_value))*cost_of_equity 
+ (debt_value/(debt_value+equity_value))*avg_cost_of_debt*(1 - tax_rate) AS WACC
FROM capital_structure
WHERE company_id = :company_id;
```
- **Prompt (RAG):** Ví dụ: “Công ty Y có nợ = 40%, vốn chủ = 60% trong tổng nguồn vốn. Chi phí vốn chủ sở hữu 12%, chi phí vay trước thuế 8%, thuế TNDN 20%. Hỏi WACC của công ty.” Hệ thống sẽ tính WACC = 0.6×12% + 0.4×8%×(1−0.2)
.


### Use Case C9 – Phân tích nợ dài hạn

- **Mục tiêu:** Phân tích cấu trúc và chi phí nợ dài hạn của doanh nghiệp, bao gồm định giá nợ bằng trái phiếu (theo market_rate) và các chỉ tiêu đòn bẩy tài chính, hệ số bao phủ lãi vay.

- **Đầu vào:** Dữ liệu nợ dài hạn gồm các trái phiếu (mệnh giá, lãi suất coupon, thời hạn), lãi suất thị trường (market_rate), dư nợ, chi phí lãi vay, các chỉ số EBITDA, khấu hao/deprec., vốn chủ sở hữu. Nếu có thuê dài hạn, dữ liệu thuê theo IFRS 16 (trái phiếu thuê, lãi vay thuê, khấu hao tài sản thuê).

- **Đầu ra:**

    - Giá trị hiện tại của các khoản nợ bằng trái phiếu (nếu áp dụng), tính theo phương pháp tương tự A14 với market_rate.

    - Tỷ lệ nợ/vốn chủ sở hữu và nợ/cơ cấu vốn (debt-to-equity, debt-to-capital).

    - Hệ số bao phủ lãi vay (Interest Coverage) = EBIT / Chi phí lãi vay. Ở đây, EBIT = EBITDA – D&A, bởi EBIT đã bao gồm khấu hao.

- **Logic tính toán:**

    - **Định giá nợ (nếu có trái phiếu):** Tương tự Use Case A14, dùng market_rate để chiết khấu các dòng tiền trái phiếu, xác định giá trị hiện tại của nợ dài hạn.

    - **Khấu hao (D&A) và EBIT:** EBIT được xác định sau khi khấu trừ D&A (chi phí khấu hao và khấu hao tài sản vô hình) từ EBITDA. Ví dụ, với EBITDA = 200 và D&A = 30, thì EBIT = 170. Lưu ý rằng nếu theo IFRS 16, chi phí thuê dài hạn được chuyển thành khấu hao tài sản thuê và chi phí lãi vay (tăng EBIT và chi phí lãi thay vì thuê hoạt động).

    - **Hệ số bao phủ lãi vay:** Tính bằng EBIT / Tổng lãi vay. Do EBIT đã bao gồm D&A, hệ số này phản ánh khả năng trả lãi thực tế. Ví dụ, EBIT 170/ chi phí lãi 50 = 3,4 lần.

    - **Tỷ lệ nợ:** Tính bằng tổng nợ dài hạn chia cho tổng vốn (nợ + vốn chủ sở hữu). IFRS 16 gia tăng nợ dài hạn (trái phiếu thuê được ghi nhận là nợ) và làm tăng EBITDA do loại bỏ chi phí thuê khỏi chi phí hoạt động. Vì vậy, áp dụng chuẩn mực mới làm tăng cả tử số và mẫu số của một số tỷ số đòn bẩy.

- **Mẫu SQL:** Ví dụ tính hệ số bao phủ và tỷ lệ nợ:
```sql
SELECT
SUM(debt_long_term) AS total_debt,
SUM(interest_expense) AS total_interest,
SUM(EBITDA) - SUM(depreciation) AS EBIT,
(SUM(EBITDA) - SUM(depreciation)) / SUM(interest_expense) AS interest_coverage,
SUM(debt_long_term) / (SUM(debt_long_term) + SUM(equity)) AS debt_to_total_capital
FROM financials
WHERE company_id = :company_id;
```
- **Prompt (RAG):** Ví dụ: “Công ty X có nợ dài hạn 500 tỷ, chi phí lãi vay 20 tỷ, EBITDA 200 tỷ, khấu hao 30 tỷ, vốn chủ sở hữu 300 tỷ. Tính tỷ lệ nợ/vốn chủ sở hữu và hệ số bao phủ lãi vay.” Hệ thống sẽ tính EBIT = 170 (=200−30) và các tỷ số tương ứng, lưu ý ảnh hưởng của khấu hao

.

### Use Case C10 — Tuân thủ chuẩn mực báo cáo (Compliance IFRS)

- **Mục tiêu nghiệp vụ:** Kiểm tra các tiêu chí tuân thủ chuẩn mực (ví dụ IFRS, GAAP). Edge cases: Cho phép lựa chọn bộ quy tắc.

- **Input:** Mã cổ phiếu, Chuẩn mực (ví dụ 'IFRS').

- **Output:** Điểm tuân thủ (% quy tắc đạt), các mục không đạt, báo cáo ngắn gọn.

- **OLAP/Data Source:** mart_master_analysis + bảng quy tắc compliance (compliance_table).

- **Logic/Tính Toán:** Đếm số tiêu chí đạt được / tổng tiêu chí. Ví dụ: IF debt_to_equity < 2 thì thỏa mãn quy định.

- **Ví dụ SQL:**
```sql
SELECT 
    CASE WHEN debt_to_equity < 2 THEN 'Compliant' ELSE 'Non-compliant' END AS de_status
FROM mart_master_analysis
WHERE symbol='VCB';
```
- **RAG Prompt:** “Kiểm tra mức độ tuân thủ IFRS của VCB, liệt kê các vấn đề cần điều chỉnh.”

### Use Case C11 — Dự báo rủi ro theo kịch bản vĩ mô

- **Mục tiêu nghiệp vụ:** Dự báo KPI (EPS, ROA, ROE…) dưới các kịch bản vĩ mô (CPI, lãi suất). Edge cases: Mô phỏng phân phối (monte carlo).

- **Input:** Các biến vĩ mô (CPI change, interest change...).

- **Output:** Dự báo KPI tương ứng; biểu đồ dự báo.

- **OLAP/Data Source:** fact_macro_timeseries (dữ liệu lịch sử vĩ mô), mart_master_analysis.

- **Logic/Tính Toán:** Tính hệ số hồi quy (đơn biến hoặc đa biến) giữa macro và KPI. Dự báo = base + slope * Δmacro. Sử dụng statsmodels hoặc numpy.polyfit.

- **Ví dụ SQL:**
```sql
SELECT m.value AS cpi, k.net_margin
FROM fact_macro_timeseries m
JOIN mart_master_analysis k ON YEAR(m.date)=k.year
WHERE m.indicator_code='VN_CPI_YOY';
```
Sau đó chạy hồi quy.

- **RAG Prompt:** “Dự báo EPS của VCB nếu CPI tăng 5%, dựa trên tương quan lịch sử.”

===========================

## D. Người Dùng Cuối / Phổ Thông (General User)
### Use Case D1 — Hỏi về khái niệm tài chính cơ bản

- **Mục tiêu nghiệp vụ:** Giải thích đơn giản các khái niệm (ví dụ "lãi suất là gì?"). Edge cases: Không dùng thuật ngữ chuyên sâu.

- **Input:** Câu hỏi (ví dụ “Lãi suất là gì?”).

- **Output:** Đoạn văn giải thích, ví dụ minh họa.

- **OLAP/Data Source:** Không cần (kiến thức chung).

- **Logic/Tính Toán:** Dựa vào kiến thức tĩnh (RRAG không cần tìm trong DB).

- **RAG Prompt:** “Hỏi Chatbot: Giải thích khái niệm lãi suất đơn giản như giải thích cho người mới.”

### Use Case D2 — “Công ty này có tốt không?” (Rating)

- **Mục tiêu nghiệp vụ:** Đưa ra điểm số tổng hợp dựa trên KPI cơ bản (0-10) cùng lời giải thích. Edge cases: Ví dụ: ROE>15% tốt, D/E>2 điểm thấp.

- **Input:** Mã cổ phiếu.

- **Output:** Điểm (0-10), các tiêu chí (ví dụ ROE, D/E) và đánh giá tương ứng (tốt/không). Narrative dễ hiểu.

- **OLAP/Data Source:** mart_master_analysis (ROE, D/E, ROA, v.v.).

- **Logic/Tính Toán:** Tính điểm tổng: ví dụ điểm = average( min(ROE/20,1), 1 - min(D/E/2,1), ... ) * 10.

- **Ví dụ SQL:**
```sql
SELECT roe_ttm, debt_to_equity
FROM mart_master_analysis
WHERE symbol='HPG' LIMIT 1;
```
- **RAG Prompt:** “Dựa trên dữ liệu, HPG có phải cổ phiếu tốt (điểm 0-10)? Lí do.”

### Use Case D3 — “Có nên đầu tư cổ phiếu X không?” (Pros/Cons)

- **Mục tiêu nghiệp vụ:** Nêu ra ưu/nhược điểm của cổ phiếu dựa trên dữ liệu (KHÔNG khuyến nghị).

- **Input:** Mã cổ phiếu.

- **Output:** Danh sách pros (ví dụ tăng trưởng cao) và cons (biến động giá lớn, nợ nhiều), ngôn ngữ dễ hiểu.

- **OLAP/Data Source:** mart_master_analysis, so sánh với trung bình ngành.

- **Logic/Tính Toán:** Lấy giá trị KPI và so sánh: ví dụ nếu growth > median tăng trưởng thì “tăng trưởng cao”. Nếu volatility > 1 thì “rủi ro biến động lớn”.

- **Ví dụ SQL:**
```sql
SELECT revenue_growth_yoy, volatility_30d
FROM mart_master_analysis
WHERE symbol='HPG';
```
- **RAG Prompt:** “Liệt kê ưu và nhược điểm của việc đầu tư vào HPG dựa trên dữ liệu tài chính.”

### Use Case D4 — Giải thích KPI tài chính đơn giản

- **Mục tiêu nghiệp vụ:** Giải thích ý nghĩa của KPI bằng ngôn ngữ đơn giản cùng ví dụ thực tế.

- **Input:** Mã cổ phiếu, KPI (ví dụ 'ROE').

- **Output:** Đoạn văn: định nghĩa ROI bằng ví dụ (ví dụ “ROE 15% nghĩa là mỗi 100đ vốn tạo ra 15đ lợi nhuận sau thuế"). Kèm giá trị thực tế.

- **OLAP/Data Source:** mart_master_analysis (giá trị KPI).

- **Logic/Tính Toán:** Lấy giá trị từ DB, viết giải thích.

- **Ví dụ SQL:**
```sql
SELECT roe_ttm
FROM mart_master_analysis
WHERE symbol='VCB';
```
- **RAG Prompt:** “Giải thích ROE của VCB cho người không chuyên. ROE VCB hiện tại là bao nhiêu?”

### Use Case D5 — So sánh hiệu quả kinh doanh giữa hai công ty bất kỳ

- **Mục tiêu nghiệp vụ:** Hiển thị nhanh so sánh các chỉ số chính giữa 2 (hoặc nhiều) ticker. Edge cases: Không quá 5-10 tickers để dễ đọc.

- **Input:** Danh sách mã cổ phiếu (2+), Kỳ.

- **Output:** Bảng so sánh: Mã, Doanh thu, Biên lợi nhuận, ROE, v.v. (các chỉ số cơ bản).

- **OLAP/Data Source:** mart_master_analysis (revenue, net_margin, roe_ttm...).

- **Logic/Tính Toán:** SELECT symbol, revenue, net_margin, roe_ttm ... WHERE symbol IN (...).

- **Ví dụ SQL:**
```sql
SELECT symbol, revenue, net_margin, roe_ttm
FROM mart_master_analysis
WHERE symbol IN ('VCB','HPG') AND year=2024;
```
- **RAG Prompt:** “So sánh VCB vs HPG: doanh thu, biên lợi nhuận, ROE.”

### Use Case D6 — Hỏi về rủi ro tài chính tổng quát

- **Mục tiêu nghiệp vụ:** Giải thích các loại rủi ro (nợ, thanh khoản) của công ty. Edge cases: Dựa trên loại yêu cầu.

- **Input:** Mã cổ phiếu, Kiểu rủi ro ('debt', 'liquidity', 'volatility'...).

- **Output:** Giải thích bằng ngôn ngữ đơn giản cùng tỷ số ví dụ. Ví dụ: "D/E của HPG là X, cao hơn ngưỡng 2 => rủi ro nợ lớn".

- **OLAP/Data Source:** mart_master_analysis (debt_to_equity, current_ratio, volatility).

- **Logic/Tính Toán:** Lấy tỷ số tương ứng, so với chuẩn ngưỡng.

- **Ví dụ SQL:**
```sql
SELECT debt_to_equity
FROM mart_master_analysis
WHERE symbol='HPG';
```
- **RAG Prompt:** “Đánh giá rủi ro tài chính của HPG: Rủi ro nợ (D/E) cao không? Giải thích.”

### Use Case D7 — Peer Comparison (Radar Chart)

- **Mục tiêu nghiệp vụ:** Xây dựng dữ liệu so sánh đa chiều (ROE, PE, growth) cho radar chart. Edge cases: Chuẩn hóa 0-100%.

- **Input:** Danh sách mã (hoặc sector).

- **Output:** JSON các chỉ số chuẩn hóa (score từ 0-100) cho mỗi công ty. Narrative chỉ ra điểm mạnh/yếu.

- **OLAP/Data Source:** mart_master_analysis (ROE, PE, revenue_growth_yoy...).

- **Logic/Tính Toán:** Cho mỗi KPI: (value - min_industry) / (max - min) * 100. Đối với mỗi symbol.

- **Ví dụ SQL:**
```sql
SELECT symbol, roe_ttm, pe_ttm, revenue_growth_yoy
FROM mart_master_analysis
WHERE sector='Banking' AND year=2024;
```
Sau đó tính normalization.

- **RAG Prompt:** “So sánh đa chiều VCB với các ngân hàng khác: radar chart các chỉ số ROE, PE, tăng trưởng.”

### Use Case D8 — Giải thích Cash Flow phân loại

- **Mục tiêu nghiệp vụ:** Giải thích cơ bản các mục CFO/CFI/CFF với ví dụ.

- **Input:** Mã cổ phiếu, Kỳ.

- **Output:** Giá trị CFO, CFI, CFF và ý nghĩa: ví dụ "CFO dương cao chứng tỏ hoạt động kinh doanh tạo ra tiền mặt."

- **OLAP/Data Source:** fact_cash_flow.

- **Logic/Tính Toán:** Lấy giá trị, check dấu +/- và tính tỷ trọng nếu cần.

- **Ví dụ SQL:**
```sql
SELECT cfo, cfi, cff
FROM fact_cash_flow
WHERE symbol='VCB' AND year=2024;
```
- **RAG Prompt:** “Giải thích dễ hiểu về CFO/CFI/CFF của VCB 2024: giá trị và ý nghĩa.”

### Use Case D9 — Hỏi đáp tương tác kèm hình ảnh

- **Mục tiêu nghiệp vụ:** Cung cấp giải thích kèm minh họa/biểu đồ. Edge cases: Chỉ câu có thể minh họa.

- **Input:** Câu hỏi (có thể yêu cầu hình ảnh).

- **Output:** Văn bản trả lời + hình ảnh đồ thị minh họa (nếu có thể). (Các hình vẽ lấy từ công cụ embed_image dựa RAG retrieval).

- **OLAP/Data Source:** Có thể truy vấn dữ liệu nếu cần vẽ chart.

- **Logic/Tính Toán:** Lấy dữ liệu, dùng công cụ vẽ/embedding để tạo ảnh (nếu hỗ trợ).

- **RAG Prompt:** “Hỏi: Giải thích ROE đơn giản cho người mới, kèm minh họa.”

## 🧠 Tích hợp vào hệ thống RAG

- **Retrieval:** Đầu tiên truy vấn OLAP (SQL) lấy số liệu, sau đó tính toán (code_execution) nếu cần (ví dụ RSI, WACC).

- **Pipeline:** Dữ liệu raw → ETL → mart/master tables → Mô-đun RAG: lấy dữ liệu và chuyển hóa thành câu trả lời.

- **Lưu ý chung:** Tránh tư vấn pháp lý hoặc khuyến nghị quyết định; tập trung vào phân tích dữ liệu khách quan.
