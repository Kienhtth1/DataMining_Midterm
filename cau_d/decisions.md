# Câu d - Data Preparation: quyết định, lý do và thay đổi của dữ liệu

`cau_d/prepare.py` dùng đầu vào là `data/04_preprocessed.csv`, ngoài ra chỉ đọc `04_change_log.csv` để dựng biến thể cho phân tích độ nhạy. Script không đọc file 01, log hay `true_segment`. Kết quả in ra nằm trong `prepare_output.txt`.

**Đầu ra:**

| File | Kích thước | Dùng để |
|---|---|---|
| `data/05_prepared.csv` | 200 × 14 | Ma trận chính cho Câu e |
| `data/05_prepared_pca.csv` | 200 × 2 | Trực quan hóa và phương án so sánh |
| `data/05_prepared_aov03.csv` | 200 × 14 | Như ma trận chính, nhưng 22 giá trị AOV lấy từ trước bước A3. Dùng cho phân tích độ nhạy đã cam kết ở Câu c |
| `cau_d/prep_params.json` | - | Toàn bộ tham số, để áp lại cho dữ liệu mới |

## Bảng quyết định

| Bước | Quyết định | Căn cứ |
|---|---|---|
| Chọn thuộc tính | Bỏ `customer_id` | Là định danh, không mang thông tin |
| | Bỏ `total_spend_12m` | Trên file 04, log(total) được giải thích bởi log(orders) + log(AOV) với R² = 0.9994. Giữ orders và AOV vẫn giữ đủ cả mức chi tiêu lẫn phong cách mua (Câu b) |
| | Bỏ `annual_income_was_missing` | Cờ này phản ánh quá trình thu thập dữ liệu, không phải hành vi. Nó tương quan với tuổi (r = -0.21, p = 0.003), nên đưa vào sẽ tạo một trục phân cụm giả |
| | Bỏ `gender`, `preferred_category` | Quy tắc: giữ thuộc tính phân loại nếu liên quan có ý nghĩa đến ít nhất 1 thuộc tính hành vi (Kruskal-Wallis, Bonferroni). `gender` không có liên quan nào (ε² ≤ 0.005). `preferred_category` có p_bonf nhỏ nhất là 0.069 |
| | Giữ `payment_method` và `city_tier` | `payment_method` liên quan đến 8/9 thuộc tính (ε² tối đa 0.19). `city_tier` liên quan đến 2/9 |
| Biến đổi | log10 cho income, orders, AOV và return_rate; log1p cho recency (có giá trị 0); giữ nguyên age, tenure, discount, session | Chỉ biến đổi khi \|skew\| > 1, và chọn cách cho \|skew\| nhỏ nhất. Ví dụ skew của orders giảm từ 4.99 xuống 0.08, của recency từ 2.00 xuống 0.04 |
| Mã hóa | `city_tier` giữ dạng số thứ bậc. `payment_method` mã hóa one-hot rồi nhân trọng số w = 1/√(1 - Σp²) = 1.166 | Trọng số này làm mỗi thuộc tính, dù là số hay định danh, có cùng đóng góp kỳ vọng (= 2) vào bình phương khoảng cách |
| Chuẩn hóa | StandardScaler **fit trên 194 khách thường**, sau đó áp dụng cho cả 200 bản ghi | So sánh 3 phương án, xem bảng dưới |
| Giảm chiều | Không dùng PCA làm đầu vào chính. Lưu PCA 2 chiều để trực quan hóa và so sánh | Xem phần PCA |
| Giảm số đối tượng | Không lấy mẫu, không loại bản ghi | n = 200, không có bản ghi trùng lặp. Lấy mẫu có thể làm mất nhóm 6 khách cực đoan (3% dữ liệu) |

### Chọn cách chuẩn hóa

Hai tiêu chí:
- **Đóng góp max/min:** tỷ số giữa đóng góp lớn nhất và nhỏ nhất của các thuộc tính, tính trên 194 khách thường. Lý tưởng là 1.
- **Khoảng cách của cặp cực đoan:** tỷ trọng khoảng cách thuộc về các cặp có ít nhất một khách cực đoan. Các cặp này chỉ chiếm 5.9% tổng số cặp.

| Phương án | Đóng góp max/min | Thuộc tính bị giảm trọng số nhất | Khoảng cách của cặp cực đoan |
|---|---|---|---|
| Standard, fit trên 200 khách | 1.76 | annual_income (var 0.59) | 13.3% |
| Robust (median, IQR/1.349) | 3.81 | city_tier (var 0.30) | 15.6% |
| **Standard, fit trên 194 khách thường** | **1.00** | - | 15.6% |

- **Robust** ước lượng sai độ phân tán của những phân bố bẹt hoặc hai đỉnh, và của `city_tier` vốn chỉ có 3 giá trị. Kết quả là trọng số chênh nhau tới 3.8 lần.
- **Standard fit trên cả 200 khách** để 6 khách cực đoan thổi phồng std của thu nhập thêm 31%, làm nén thu nhập của 194 khách còn lại.
- **Phương án được chọn** cho đóng góp đều tuyệt đối với khách thường, còn 6 khách cực đoan vẫn ở xa. Điều đó đúng, vì họ thật sự khác biệt. Cái giá là phần khoảng cách của nhóm cực đoan cao hơn một chút (15.6% so với 13.3%).

### PCA

| K | Phương sai giữ lại | Tương quan khoảng cách cặp | Tỷ lệ giữ 10 láng giềng gần |
|---|---|---|---|
| 2 (phân tích song song) | 53.3% | 0.880 | 28.5% |
| 3 | 63.6% | 0.930 | 40.6% |
| 6 (80% phương sai) | 82.4% | 0.977 | 58.4% |
| 9 | 92.7% | 0.993 | 74.6% |

Phân tích song song chỉ giữ 2 thành phần mạnh hơn nhiễu:
- **PC1** là trục "giá trị khách hàng": income 0.55, AOV 0.46, orders 0.38.
- **PC2** là trục "mức tương tác và săn khuyến mãi": session 0.55, orders 0.43, discount 0.42, recency -0.37.

Trên mặt phẳng PC1-PC2 thấy rõ 3 nhóm, với 6 khách cực đoan nằm ở rìa (`d_pca.png`). Tuy vậy PCA 2 chiều chỉ giữ được 28% láng giềng gần. 14 chiều với n = 200 không phải bài toán chiều cao, nên giảm chiều không cần thiết và còn làm mất cấu trúc cục bộ mà phân cụm cần. Vì vậy dữ liệu chính vẫn ở đủ 14 chiều. PCA dùng để vẽ hình và làm phương án đối chứng ở Câu e.

## Dữ liệu thay đổi thế nào

- **Đóng góp vào khoảng cách** (`d_feature_contribution.png`):
  - Dữ liệu thô: `total_spend_12m` chiếm gần 100%.
  - Chỉ chuẩn hóa z-score: các thuộc tính chia đều, nhưng mức chi tiêu bị tính hai lần và chưa có thuộc tính phân loại.
  - Sau chuẩn bị: 11 thuộc tính đóng góp 8.0-14.1% mỗi thuộc tính trên cả 200 khách. Riêng income cao hơn (14.1%) do có 6 khách cực đoan.
- **Ảnh hưởng của nhóm cực đoan:** với dữ liệu chỉ chuẩn hóa z-score, các cặp có khách cực đoan chiếm 36.7% tổng bình phương khoảng cách. Sau chuẩn bị chỉ còn 14.8%: họ vẫn ở xa nhưng không còn lấn át.
- **Độ lệch:** |skew| lớn nhất giảm từ 5.04 xuống 1.50. Riêng income vẫn còn 1.50 vì đuôi phải là 6 khách cực đoan; đây là thông tin thật, không phải lỗi.
- **Tương quan:** sau khi bỏ total, |Pearson| lớn nhất giữa các thuộc tính số là 0.72 (income - AOV). Không còn cặp nào gần như trùng lặp.
- **Hopkins** (0.5 là ngẫu nhiên, càng gần 1 càng có cấu trúc cụm):
  - chỉ chuẩn hóa z-score: 0.84
  - sau chuẩn bị: 0.66
  - PCA 2 chiều: 0.80

  Giá trị giảm sau chuẩn bị **không** có nghĩa là cấu trúc cụm kém đi. Ở dữ liệu chỉ chuẩn hóa z-score, vài điểm cực đoan kéo giãn khung lấy mẫu đều, làm Hopkins cao giả tạo. Mức 0.66 vẫn cao hơn rõ rệt so với 0.5, nên dữ liệu có xu hướng cụm. Câu e sẽ kiểm chứng điều này trực tiếp bằng kết quả phân cụm.

## Gợi ý cho Câu e

- Dữ liệu có cả thuộc tính số lẫn one-hot đã cân trọng số, nên dùng được ngay với K-means và Agglomerative (Ward). Với DBSCAN, cần chọn eps theo đồ thị k-distance.
- Ba phép so sánh nên làm:
  1. `05_prepared` so với `05_prepared_pca`: ảnh hưởng của giảm chiều.
  2. `05_prepared` so với `05_prepared_aov03`: độ nhạy với bước A3 ở Câu c.
  3. Dữ liệu trước xử lý (03) so với sau xử lý: đúng yêu cầu của đề.
