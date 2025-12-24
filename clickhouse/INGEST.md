# INGESTION GUIDELINE 

### 1. Bảng: `dim_company`

**Mục đích:** Danh sách công ty, ngành nghề, sàn giao dịch.

* **Nguồn dữ liệu:** `vnstock` (Hàm `listing_companies()`).
* **Chiến lược:** Full Load (Tải lại toàn bộ hàng tuần).

**Logic Ingestion & Processing:**

1. **Python (Bronze):** Gọi API lấy danh sách toàn bộ mã chứng khoán.
2. **Logic:**
* Lọc sàn: Chỉ lấy HOSE, HNX, UPCOM.
* Map cột: `ticker` -> `symbol`, `comGroupCode` -> `exchange`.
* Tạo `UUIDv7` cho mỗi công ty.
* Lưu thẳng vào ClickHouse (vì dữ liệu nhỏ, < 2000 dòng).



```python
# Code mẫu (Python)
from vnstock import listing_companies
from uuid6 import uuid7
# ... code kết nối clickhouse ...

df = listing_companies()
df['company_id'] = [str(uuid7()) for _ in range(len(df))]
df = df.rename(columns={'ticker': 'symbol', 'organName': 'company_name_vn', ...})
client.insert_df('dim_company', df[['symbol', 'company_id', 'company_name_vn', ...]])

```

---

### 2. Bảng: `dim_period`

**Mục đích:** Bảng chuẩn hóa thời gian (Quý/Năm) để join báo cáo tài chính.

* **Nguồn dữ liệu:** **Không cần nguồn ngoài**. Tự sinh bằng SQL.
* **Chiến lược:** Chạy 1 lần duy nhất (One-time generation).

**Logic Processing:**

* Chạy câu lệnh SQL sau trực tiếp trong ClickHouse để tạo dữ liệu cho 20 năm (2015-2035).

```sql
INSERT INTO dim_period (period_id, year, quarter, period_type, start_date, end_date)
SELECT
    generateUUIDv7(),
    toYear(d),
    toQuarter(d),
    'Q',
    toStartOfQuarter(d),
    toEndOfQuarter(d)
FROM (SELECT toDate('2015-01-01') + INTERVAL number MONTH AS d FROM numbers(240))
WHERE toMonth(d) % 3 = 1; -- Lấy tháng đầu mỗi quý

```

---

### 3, 4, 5. Bảng: `fact_income_statement`, `fact_balance_sheet`, `fact_cash_flow`

**Mục đích:** Báo cáo tài chính (Kết quả kinh doanh, Cân đối kế toán, Lưu chuyển tiền tệ).
**Thách thức:** Dữ liệu trả về dạng "Ngang" (Các quý là cột), cần xoay dọc.

* **Nguồn dữ liệu:** `vnstock` (Hàm `financial_report`).
* **Chiến lược:** Incremental Load (Tải quý mới nhất).

**Logic Ingestion (Bronze - Python):**

* Tải Raw Data về dưới dạng file Parquet. Lưu ý đặt tên file theo mã và loại báo cáo.
* Ví dụ: `./datalake/bronze/financials/IncomeStatement/HPG.parquet`



**Logic Processing (Silver - Spark):**

1. **Đọc (Read):** Spark đọc file Parquet.
2. **Unpivot (Xoay bảng):** Chuyển các cột `2024-Q3`, `2024-Q2` thành dòng.
* Từ: `Symbol | Revenue | 2024-Q3 | 2024-Q2`
* Thành: `Symbol | Revenue | Period | Value`


3. **Mapping:** Đổi tên tiếng Việt sang tiếng Anh theo schema của bạn.
* `"Doanh thu thuần"` -> `revenue`
* `"Giá vốn hàng bán"` -> `cogs`


4. **Tạo ID:** Tạo `period_id` (UUID) khớp với bảng `dim_period`.
5. **Ghi (Write):** Ghi vào bảng tương ứng trong ClickHouse (`fact_income_statement`, v.v.).

---

### 6. Bảng: `fact_daily_market`

**Mục đích:** Dữ liệu giá (OHLCV) và Khối lượng. Đây là bảng nặng nhất.

* **Nguồn dữ liệu:** `vnstock` (Hàm `stock_historical_data`).
* **Chiến lược:** Daily Batch (Chạy cuối ngày).

**Logic Ingestion (Bronze - Python):**

* Vòng lặp `for` qua danh sách mã trong `dim_company`.
* Tải dữ liệu từ năm 2010 đến nay (lần đầu) hoặc ngày hôm nay (hàng ngày).
* **Quan trọng:** Lưu thành file Parquet phân vùng theo ngày.
* Path: `./datalake/bronze/market/date=2024-12-25/HPG.parquet`



**Logic Processing (Silver - Spark):**

1. **Clean:** Ép kiểu dữ liệu (`Close` -> Float, `Volume` -> Long).
2. **Fill Null:** Điền 0 cho các giá trị null (nếu có).
3. **Calculate:**
* Tính `value` (Giá trị giao dịch) = `close` * `volume` (nếu nguồn thiếu).
* Tính `market_cap` = `close` * `shares_outstanding` (Join với `dim_company`).


4. **Lưu ý cột Margin/Foreign:**
* Cột `foreign_buy`/`sell`: Có sẵn trong `vnstock`.
* Cột `margin_ratio`: **Không có miễn phí**. Gán mặc định = 0 hoặc xóa cột này khỏi schema.



---

### 6.1. Bảng: `fact_risk_metrics`

**Mục đích:** Chứa Beta, Volatility (Biến động), Alpha.

* **Nguồn dữ liệu:** **Được tính toán** từ `fact_daily_market`.
* **Công cụ:** Spark (Xử lý cửa sổ trượt - Window Functions).

**Logic Processing (Silver - Spark):**

1. **Input:** Load dữ liệu giá của Cổ phiếu và `VNINDEX`.
2. **Transform:** Tính % Thay đổi giá hàng ngày (Daily Return).
3. **Join:** Ghép bảng Cổ phiếu với `VNINDEX` theo ngày.
4. **Window Calc:**
* **Beta:** Dùng công thức hiệp phương sai (Covariance) trên cửa sổ 1 năm (252 ngày giao dịch).
* **Volatility:** Dùng hàm `stddev` (độ lệch chuẩn) trên cửa sổ 30 ngày.


5. **Output:** Ghi kết quả vào bảng `fact_risk_metrics`.

---

### 7. Bảng: `dim_macro_indicator`

**Mục đích:** Danh sách các chỉ số vĩ mô (CPI, GDP, Lãi suất).

* **Nguồn dữ liệu:** `wbdata` (World Bank) hoặc nhập tay.
* **Chiến lược:** Nhập thủ công hoặc Script đơn giản.

**Logic Ingestion:**

* Tạo một file CSV/Excel chứa danh sách các chỉ số bạn quan tâm (Mã, Tên, Đơn vị, Nguồn).
* Dùng Python đọc file này và `INSERT` vào ClickHouse.
* Ví dụ: `VN_CPI_YOY`, `Vietnam CPI Year-over-Year`, `%`, `GSO`.

---

### 8. Bảng: `fact_macro_timeseries`

**Mục đích:** Dữ liệu lịch sử của các chỉ số vĩ mô.

* **Nguồn dữ liệu:** `wbdata` (World Bank API) cho dữ liệu năm/quý. Tổng cục thống kê (GSO) cho dữ liệu tháng.
* **Chiến lược:** Python Script.

**Logic Ingestion:**

1. Dùng thư viện `wbdata` để lấy GDP, Lạm phát của Việt Nam (Mã nước: `VNM`).
2. Chuẩn hóa về dạng: `indicator_code` | `date` | `value`.
3. Insert vào ClickHouse.

---

### 9. Bảng: `fact_bond_data`

**Mục đích:** Dữ liệu trái phiếu (Dùng để tính lãi suất phi rủi ro - Risk Free Rate).

* **Nguồn dữ liệu:** Không có nguồn free cho Trái phiếu doanh nghiệp chi tiết.
* **Giải pháp:** Chỉ theo dõi **Trái phiếu Chính phủ 10 năm**.

**Logic Ingestion (Python Scraper):**

1. Viết script dùng `BeautifulSoup` hoặc `Selenium` cào nhẹ trang *TradingEconomics* hoặc *Investing.com* lấy "Vietnam 10Y Bond Yield".
2. Gán `symbol` = 'VN_GOV_10Y'.
3. Lưu vào bảng. Các cột chi tiết khác (Coupon, Maturity) để null hoặc gán giá trị giả lập.

---

### 10. Bảng: `fact_forecast`

**Mục đích:** Dự báo doanh thu/lợi nhuận tương lai.

* **Nguồn dữ liệu:** Không có API free cho dự báo của chuyên gia (Analyst Consensus).
* **Giải pháp:** **Dữ liệu người dùng tự nhập**.

**Logic Ingestion:**

* Tạo một file Excel/CSV mẫu: `Symbol | Year | Scenario | Revenue_Forecast | Profit_Forecast`.
* Bạn tự điền dự đoán của mình vào.
* Dùng Python đọc file Excel này và đẩy vào ClickHouse. Đây là cách tốt nhất để mô phỏng tính năng "Analysis" mà không tốn tiền mua dữ liệu.

---

### 11. Bảng: `fact_budget`

**Mục đích:** Kế hoạch kinh doanh do công ty công bố đầu năm.

* **Nguồn dữ liệu:** Web Scraper (CafeF/Vietstock thường có bài viết tóm tắt) hoặc nhập tay.
* **Giải pháp:** Nhập tay từ tài liệu ĐHĐCĐ (Đại hội đồng cổ đông).

**Logic Ingestion:**

* Tương tự `fact_forecast`. Vì số liệu này mỗi năm chỉ có 1 lần cho mỗi công ty, bạn nên nhập tay vào file CSV và load vào DB.

---

### 12. Bảng: `fact_sector_benchmark`

**Mục đích:** Chỉ số trung bình ngành (PE ngành, ROE ngành).

* **Nguồn dữ liệu:** **Tự động tính toán** từ dữ liệu có sẵn.
* **Cơ chế:** `Materialized View` (AggregatingMergeTree).

**Logic Processing:**

* Bạn **không cần viết code ingestion** cho bảng này.
* Trong schema SQL, bạn đã định nghĩa `CREATE MATERIALIZED VIEW mv_sector_benchmark`.
* Ngay khi dữ liệu được nạp vào `mart_master_analysis`, ClickHouse sẽ tự động tính toán trung bình ngành và điền vào bảng này.

---

### 13. Bảng: `mart_master_analysis`

**Mục đích:** Bảng tổng hợp (OLAP) dùng để vẽ biểu đồ và phân tích cuối cùng.

* **Nguồn dữ liệu:** **Tự động tổng hợp** từ tất cả các bảng trên.
* **Cơ chế:** `Materialized View`.

**Logic Processing:**

* Đây là bảng đích đến. Logic SQL cực kỳ phức tạp trong câu lệnh `CREATE MATERIALIZED VIEW mv_master_analysis` sẽ tự động chạy (Trigger) mỗi khi có dữ liệu mới ở các bảng Fact (Income, Market, Balance Sheet).
* **Lưu ý quan trọng:** Khi nạp dữ liệu lịch sử (10 năm), hãy dùng lệnh `DETACH TABLE mv_master_analysis` để tắt tạm thời view này, nạp xong dữ liệu gốc thì `ATTACH` lại để tránh làm chậm hệ thống.

---

### Tóm tắt quy trình chạy (Pipeline Execution Order)

Để hệ thống hoạt động, bạn chạy theo thứ tự sau:

1. **Setup:** Chạy SQL tạo bảng `dim_period`.
2. **Bronze (Python):**
* Chạy `ingest_companies.py` -> Nạp `dim_company`.
* Chạy `ingest_market_prices.py` -> Tạo file Parquet giá.
* Chạy `ingest_financials.py` -> Tạo file Parquet BCTC.


3. **Silver (Spark):**
* Chạy `spark_process_market.py` -> Nạp `fact_daily_market`.
* Chạy `spark_process_financials.py` -> Nạp 3 bảng BCTC.
* Chạy `spark_calc_risk.py` -> Nạp `fact_risk_metrics`.


4. **Manual/Hack:**
* Chạy script nạp Macro, Forecast (từ Excel/CSV).


5. **Gold (Auto):**
* Kiểm tra `mart_master_analysis`, dữ liệu sẽ tự động xuất hiện.