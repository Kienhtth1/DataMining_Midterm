# Câu b - Data Understanding: các vấn đề chính của dữ liệu

Nguồn: `cau_b/eda.py` chạy trên `data/03_raw_final.csv` (200 bản ghi, 14 thuộc tính: 10 số, 1 thứ bậc, 3 định danh). Không dùng `injection_log`, file 01 hay `true_segment`. Mọi con số dưới đây có trong `cau_b/eda_output.txt`.

## Tóm tắt

| # | Vấn đề | Bằng chứng chính | Ảnh hưởng đến phân cụm | Hướng xử lý (Câu c, d) |
|---|---|---|---|---|
| 1 | Lệch phải mạnh và thang đo chênh nhau tới 10⁹ lần | skew đến 6.43; std total / std return_rate = 1.04e9 | Khoảng cách Euclid gần như chỉ phụ thuộc `total_spend_12m` | log10 cho thuộc tính lệch phải, sau đó chuẩn hóa |
| 2 | Hai loại ngoại lai khác bản chất: 7 lỗi chắc chắn và 6 khách số đơn cực đoan nhưng nhất quán | 7 vi phạm quy tắc; 6 bản ghi orders 144-250, tỷ lệ total/(orders×AOV) = 1.00 | Lỗi kéo lệch tâm cụm và scaler; xóa nhóm cực đoan thì mất một phân khúc thật | Lỗi → coi là thiếu rồi điền; giữ nhóm cực đoan |
| 3 | 27% bản ghi có ô thiếu; `annual_income` thiếu phụ thuộc tuổi (MAR) | 32.0% / 11.4% / 4.7% theo nhóm tuổi, p xu hướng = 0.00018 | Xóa theo dòng làm lệch mẫu về khách lớn tuổi; điền trung bình làm sai thu nhập của khách trẻ | Điền có điều kiện theo tuổi (KNN/hồi quy) |
| 4 | `total_spend_12m` dư thừa hoàn toàn: là tích của `orders_12m` và AOV | Hồi quy log(total) ~ log(orders) + log(AOV): R² = 0.997, hệ số 0.999 và 0.988; VIF(total) = 388 | Chiều "mức chi tiêu" bị tính thêm một lần trong khoảng cách | Giữ log(orders) và log(AOV), bỏ total; KHÔNG làm ngược lại |

Vấn đề phụ: dữ liệu hỗn hợp kiểu (cần mã hóa), giá trị dồn ở biên (`discount` = 0 hoặc 1, `recency` = 365).

---

## Vấn đề 1 - Phân bố lệch phải mạnh, thang đo chênh nhau rất lớn

**Bằng chứng** (mục 3, `b_distributions.png`):
- Độ lệch (skew) của các thuộc tính tiền và số lượng rất cao: `total_spend_12m` 6.43, `orders_12m` 4.99, `annual_income` 4.64, `avg_order_value` 3.11, `recency_days` 1.99. Giá trị lớn nhất của `total_spend_12m` gấp 347 lần trung vị.
- Sau khi lấy log10, skew giảm còn 0.08 (orders), 0.66 (AOV), 0.91 (total) và 1.39 (income). Biến đổi log đưa phần lớn các thuộc tính này về gần đối xứng; riêng `annual_income` vẫn còn lệch.
- Độ lệch chuẩn của `total_spend_12m` là 1.28e8 VND, còn của `return_rate` là 0.124. Tỷ số giữa hai độ lệch chuẩn này là **1.04 × 10⁹**.
- Trên thang log, `orders_12m` và `total_spend_12m` có **nhiều đỉnh**: một nhóm khoảng 1-6 đơn, một nhóm khoảng 10-40 đơn, và vài điểm tách xa ở trên 100 đơn. Đây là dấu hiệu tích cực cho thấy dữ liệu có cấu trúc cụm.

**Ảnh hưởng đến phân cụm:**
- K-means và các thuật toán dựa trên khoảng cách Euclid, nếu chạy trên dữ liệu thô, sẽ thực chất chỉ phân cụm theo `total_spend_12m` (và AOV). Các thuộc tính hành vi như `discount_usage_rate` hay `return_rate` gần như không có trọng số.
- Phần đuôi dài kéo tâm cụm về phía các điểm cực đại. Đa số khách sẽ bị dồn vào một cụm lớn, cấu trúc nhiều đỉnh nhìn thấy trên thang log không được phản ánh.
- Chuẩn hóa z-score áp lên dữ liệu chưa log vẫn bị phần đuôi chi phối. Vì vậy phải lấy log trước rồi mới chuẩn hóa.

## Vấn đề 2 - Ngoại lai gồm hai loại khác bản chất; IQR một mình không phân biệt được

**Bằng chứng** (mục 2, mục 6, mục 7; `b_outliers_iqr.png`, `b_orders_vs_aov.png`):

1. **7 bản ghi vi phạm miền giá trị.** Đây là lỗi chắc chắn, vì các giá trị này không thể xảy ra:

   | customer_id | Thuộc tính | Giá trị |
   |---|---|---|
   | C0019 | age | 150 |
   | C0191 | age | 3 |
   | C0120 | tenure_months | -5 |
   | C0062 | avg_order_value | -566 000 |
   | C0161 | recency_days | 9999 |
   | C0116 | return_rate | 1.8 |
   | C0166 | total/(orders×AOV) | 1003, tức sai đơn vị khoảng ×1000 |

2. **6 bản ghi có `orders_12m` cực đoan** (C0052, C0093, C0112, C0126, C0147, C0172) **nhưng nhất quán nội tại**:
   - số đơn từ 144 đến 250, trong khi phần còn lại tối đa 41 (khoảng trống giữa 41 và 144);
   - tỷ lệ total/(orders×AOV) đúng bằng 1.00 ở cả 6 bản ghi;
   - thu nhập 1 765-3 338 triệu/năm;
   - mua gần đây, recency 0-4 ngày.

   Các thuộc tính "kể cùng một câu chuyện" nên đây nhiều khả năng là một nhóm khách thật, ví dụ khách mua sỉ.
3. **`avg_session_minutes` = 1440 phút/tuần.** Giá trị này không vi phạm miền, vì 1440 phút/tuần tương đương khoảng 3.4 giờ/ngày, vẫn có thể xảy ra. Tuy vậy nó lớn gấp 13 lần ngưỡng IQR trên là 113. Đây là ngoại lai đáng ngờ chứ chưa phải lỗi chắc chắn.
4. **Quy tắc IQR trên thang thô gắn cờ 71/200 bản ghi (35.5%).** Phần lớn là đuôi tự nhiên của các phân bố lệch: 45 bản ghi ở `total_spend_12m` và 30 ở AOV. Ngược lại, IQR trên thang log10 **không bắt được** 6 bản ghi orders cực đoan (0 bản ghi), vì thang log nén phần đuôi. Như vậy kết quả phát hiện ngoại lai phụ thuộc mạnh vào thang đo, không thể dùng máy móc một quy tắc.

**Ảnh hưởng đến phân cụm** (mục 7):
- Một giá trị lỗi đủ làm méo thống kê. Chỉ riêng giá trị 1440 đã làm độ lệch chuẩn của `avg_session_minutes` tăng từ 22.1 lên 105.9. Khi chuẩn hóa z-score, cả thuộc tính này bị ép lại gần 0 và mất vai trò trong việc phân cụm.
- Pearson(orders, session) bằng 0.036 trên dữ liệu đủ, tăng lên 0.178 khi bỏ giá trị 1440, và lên 0.630 khi bỏ thêm 6 bản ghi orders cực đoan. Spearman thì là 0.652 ngay từ đầu. Các phương pháp dựa trên tuyến tính và phương sai như K-means hay PCA sẽ "không thấy" một quan hệ thật nếu không xử lý ngoại lai.
- Nếu xóa 6 bản ghi orders cực đoan, mô hình mất một phân khúc thật, nhỏ nhưng có giá trị kinh doanh cao. Nếu giữ nguyên mà không biến đổi, K-means có thể dành một cụm cho riêng 6 điểm này, hoặc chúng kéo tâm cụm VIP lệch đi.
- Nếu xóa tất cả 71 bản ghi bị IQR gắn cờ thì mất 35% dữ liệu và cắt bỏ chính nhóm VIP.

## Vấn đề 3 - 27% bản ghi có ô thiếu; `annual_income` thiếu theo cơ chế MAR phụ thuộc tuổi

**Bằng chứng** (mục 5, `b_missing.png`):
- Có 3 thuộc tính bị thiếu: `annual_income` 31 ô (15.5%), `avg_session_minutes` 16 ô (8.0%), `city_tier` 10 ô (5.0%). Tổng cộng 54 bản ghi (27.0%) có ít nhất một ô thiếu.
- Tỷ lệ thiếu `annual_income` giảm dần theo tuổi:

  | Nhóm tuổi | Thiếu / tổng | Tỷ lệ |
  |---|---|---|
  | 18-27 | 16/50 | 32.0% |
  | 28-39 | 12/105 | 11.4% |
  | 40+ | 2/43 | 4.7% |

  Kiểm định xu hướng Cochran-Armitage cho z = -3.75, p = 0.00018. Trung vị tuổi của nhóm thiếu là 27, của nhóm không thiếu là 34.
- `city_tier` (p = 0.43) và `avg_session_minutes` (p = 0.17) không có xu hướng theo tuổi, cũng không khác biệt với các thuộc tính khác. Kết quả này phù hợp với MCAR, nhưng cỡ mẫu nhỏ nên lực kiểm định thấp.
- **Lưu ý trung thực:** ở bước sàng lọc rộng gồm 39 phép thử, cặp (thiếu income, age) có p = 0.0035 nhưng sau hiệu chỉnh Bonferroni p = 0.138, không còn ý nghĩa. Kết luận MAR dựa vào kiểm định xu hướng có định hướng ở bước 2 và mức độ chênh lệch lớn giữa các nhóm (32% so với 4.7%). Từ dữ liệu quan sát được thì **không loại trừ được MNAR**, tức là thiếu do chính mức thu nhập (ví dụ người thu nhập thấp không khai). Nhóm thiếu income có trung vị AOV thấp hơn (3.45e5 so với 4.76e5) nhưng khác biệt không có ý nghĩa (p = 0.16).

**Ảnh hưởng đến phân cụm** (mục 7):
- Xóa theo dòng chỉ còn lại 146/200 bản ghi, và tỷ lệ khách 18-27 tuổi giảm từ 25.0% xuống 19.9%. Các cụm sẽ thiếu khách trẻ, đúng nhóm có hành vi khác biệt: tỷ lệ dùng mã giảm giá tương quan âm với tuổi, ρ = -0.40.
- Điền giá trị thiếu bằng trung bình hoặc trung vị chung sẽ gán cho khách trẻ mức thu nhập của toàn bộ mẫu, cao hơn thực tế vì ρ(age, income) = 0.40. Điều này tạo ra một dải điểm giả nằm giữa các cụm.
- Vì tính hợp lệ của việc điền có điều kiện dựa trên giả định MAR, cần thêm một **cờ thiếu** (missing indicator) hoặc phân tích độ nhạy để phòng trường hợp thực ra là MNAR.

## Vấn đề 4 - `total_spend_12m` dư thừa vì là tích của số đơn và giá trị đơn

**Tại sao tương quan từng cặp không đủ.** Định nghĩa total ≈ orders × AOV là quan hệ **nhân**. Tương quan từng cặp chỉ đo được quan hệ của total với từng thành phần riêng lẻ: ρ(orders, total) = 0.94 cao, nhưng ρ(AOV, total) chỉ 0.45. Nhìn riêng hai con số này dễ kết luận sai rằng AOV "ít trùng lặp" với total. Mức dư thừa thật phải đo bằng việc total được giải thích bởi **cả hai thành phần cùng lúc**.

**Bằng chứng** (mục 4b, `b_spend_redundancy.png`, n = 198 sau khi bỏ các ô vi phạm miền):
- Hồi quy log10(total) = 0.074 + **0.999**·log10(orders) + **0.988**·log10(AOV) có **R² = 0.9974**. Phần dư có độ lệch chuẩn 0.038 (log10), tương đương khoảng ±9% trên thang gốc, có thể do làm tròn hoặc nhiễu đo lường. Hai hệ số xấp xỉ 1 đúng như quan hệ tích.
- Mỗi thành phần đứng riêng chỉ giải thích được một phần: R² = 0.819 với orders, 0.537 với AOV.
- VIF trên thang log: total 388, orders 210, AOV 82. Cả ba đều vượt xa ngưỡng 10, tức là đa cộng tuyến hoàn toàn. Trong khi đó ρ(orders, AOV) = 0.25, nên hai thành phần gần độc lập và mỗi cái mang thông tin riêng.

**Ảnh hưởng đến phân cụm.** Sau khi lấy log và chuẩn hóa, log(total) gần như là tổng của log(orders) và log(AOV). Giữ cả ba thì hướng "chi tiêu tổng" được cộng trọng số thêm một lần trong khoảng cách Euclid, còn các chiều hành vi khác bị giảm vai trò tương ứng.

**Bỏ thuộc tính nào: không thể bỏ ngược lại (chỉ giữ total).** Không gian log(orders) × log(AOV) có hai chiều:
- **mức chi tiêu**, tức log(total) = log(orders) + log(AOV);
- **phong cách mua**, tức style = log(orders) - log(AOV): nhiều đơn rẻ hay ít đơn đắt.

Chỉ giữ total là chiếu mọi điểm lên trục thứ nhất và đánh mất trục thứ hai. Trục thứ hai mang thông tin hành vi riêng:

| Hành vi | ρ với log(total) | ρ với style |
|---|---|---|
| discount_usage_rate | **-0.03** | **0.49** |
| avg_session_minutes | 0.59 | 0.76 |
| recency_days | -0.54 | -0.49 |
| return_rate | -0.45 | -0.16 |

- Tỷ lệ dùng mã giảm giá **không liên quan** đến tổng chi tiêu nhưng liên quan rõ đến phong cách mua. Khách săn khuyến mãi được nhận diện bằng "nhiều đơn rẻ", không phải bằng "chi nhiều hay ít".
- Trong 867 cặp khách có tổng chi tiêu chênh nhau dưới 10%, có 72 cặp (8.3%) có tỷ số orders/AOV chênh nhau trên 10 lần. Ví dụ C0016 mua 2 đơn × 1.14 triệu = 2.39 triệu, còn C0097 mua 16 đơn × 144 nghìn = 2.27 triệu. Nếu chỉ dùng total, hai khách này gần như trùng nhau, trong khi hành vi hoàn toàn khác.

**Kết luận cho Câu d:** giữ log(orders_12m) và log(avg_order_value), bỏ total_spend_12m. Cách này không mất thông tin, vì R² = 0.997 cho thấy total được tái tạo gần như hoàn toàn từ hai thuộc tính kia, đồng thời loại bỏ phần trọng số bị tính trùng. Một cách tương đương là giữ log(total) cùng style, vì đó chỉ là phép xoay của cùng hai trục.

## Vấn đề phụ

- **Dữ liệu hỗn hợp kiểu.** Có `city_tier` (thứ bậc, 1-3) và 3 thuộc tính định danh: `gender` (2 giá trị), `preferred_category` (5) và `payment_method` (4). K-means không xử lý trực tiếp được kiểu dữ liệu này. Mã hóa one-hot sẽ thêm khoảng 8-11 cột nhị phân, dễ lấn át 10 thuộc tính số. Có thể chọn k-prototypes hoặc khoảng cách Gower, hoặc chỉ mã hóa có chọn lọc.
- **Giá trị dồn ở biên.** `discount_usage_rate` có 3 giá trị đúng bằng 0 và 4 giá trị đúng bằng 1; `recency_days` có 6 giá trị đúng bằng 365 và 14 giá trị bằng 0. Với một tỷ lệ liên tục, việc dồn đúng ở biên có thể là dấu hiệu của cắt biên hoặc nhiễu. Với recency, giá trị 365 là giới hạn của cửa sổ quan sát 12 tháng, nên đây là dữ liệu bị cắt phải (right-censored). Ảnh hưởng nhỏ, nhưng nên ghi nhận.

## Hạn chế của phân tích này

- Nhiễu ngẫu nhiên không phát hiện được trực tiếp từng ô; mới chỉ thấy dấu hiệu gián tiếp như dồn biên.
- Các kiểm định cơ chế thiếu chạy trên 200 bản ghi, nên lực kiểm định với MCAR thấp.
- Các quy tắc miền giá trị (tuổi 18-100, recency ≤ 365...) là giả định nghiệp vụ do người phân tích đặt ra, cần được ghi rõ.
