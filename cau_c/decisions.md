# Câu c - Tiền xử lý: quyết định, lý do và đánh giá

- `cau_c/preprocess.py` chỉ đọc `data/03_raw_final.csv`. Kết quả in ra nằm trong `preprocess_output.txt`.
- `cau_c/evaluate.py` là file **duy nhất** đọc file 01 và `injection_log`, chạy sau khi tiền xử lý đã xong. Kết quả nằm trong `evaluate_output.txt`.
- Đầu ra:
  - `data/04_preprocessed.csv`: 200 × 16, gồm 15 cột gốc và cờ `annual_income_was_missing`.
  - `data/04_change_log.csv`: 87 ô thay đổi, mỗi ô ghi giá trị cũ, giá trị mới, bước và lý do.
  - `cau_c/imputed_variants.csv`: giá trị điền của cả 3 cách cho mọi ô.

## Bảng quyết định

| Bước | Vấn đề (từ Câu b) | Quyết định | Lý do | Kết quả so với file 01 |
|---|---|---|---|---|
| A1 | C0166: total/(orders×AOV) ≈ 1003 | total = orders × AOV | Tỷ lệ ≈ 10³ cho thấy lỗi đơn vị nằm ở total. Đây là quan hệ xác định nên không cần ước lượng thống kê | Sai số 0.3% |
| A1 | C0062: AOV = -566 000 | AOV = total / orders = 589 000 | \|AOV\| lệch 3.9% so với total/orders, khả năng là lỗi dấu | Sai số 4.1% |
| A1 | 5 ô vi phạm miền: tuổi 150 và 3, tenure -5, recency 9999, return_rate 1.8 | Đổi thành NaN rồi điền ở A5 | Không có quan hệ xác định để khôi phục. Đổi riêng ô lỗi, không xóa cả bản ghi | Sai số 8-55%, xem E2 |
| A2 | C0195: avg_session_minutes = 1440 | Coi là lỗi nghi ngờ, đổi thành NaN rồi điền bằng KNN | 1440 = 24 × 60, đúng bằng số phút của một ngày nên nghi nhập nhầm đơn vị. Giá trị này gấp 12.8 lần ngưỡng IQR, trong khi các khách có 15-21 đơn có session tối đa 80.6 | Điền được 45.5, giá trị thật 47.4 (sai số 4%) |
| A3 | Nhiễu ở AOV | AOV = total / orders cho 22 bản ghi có tỷ lệ nằm ngoài dải IQR log [0.880, 1.136] | Theo định nghĩa, AOV là trung bình giá trị đơn | Precision 1.00, recall 0.37. Trên 22 bản ghi được sửa, sai số giảm từ 20.5% xuống 2.8% |
| A3 | Nhiễu ở session và discount | Không sửa từng ô | Không có quan hệ xác định để đối soát | Sai số tương đối trung bình vẫn khoảng 22-23% trên các ô nhiễu |
| A5 | Giá trị thiếu và các ô lỗi | Chọn cách điền riêng cho từng cột bằng kiểm định chéo nội bộ trên file 03 | Không cần nhìn đáp án vẫn chọn được cách điền | Xem E1 |
| - | 6 khách có số đơn cực đoan | Giữ nguyên | Các thuộc tính nhất quán nội tại (Câu b) | 0 ô quan sát bị đổi; 1 ô thiếu được điền |

## Điền giá trị thiếu: so sánh 3 cách

**Ba cách điền:**
- **Trung vị:** dùng trung vị của từng cột.
- **KNN:** 5 láng giềng, trọng số theo khoảng cách.
- **Iterative:** hồi quy BayesianRidge theo vòng lặp.

KNN và Iterative chạy trên không gian đã biến đổi: log10 cho income, orders và AOV; log1p cho recency và session; sau đó chuẩn hóa z-score. `total_spend_12m` bị loại khỏi tập thuộc tính dùng để điền vì dư thừa (Câu b, R² = 0.997).

**Lý do so sánh như vậy:** `annual_income` thiếu theo cơ chế MAR phụ thuộc tuổi (Câu b). Điền bằng trung vị chung sẽ gán cho khách trẻ một mức thu nhập quá cao. KNN và Iterative tận dụng được các tương quan ρ(age, income) = 0.40 và ρ(income, AOV) = 0.63.

**Cách chọn mà không nhìn đáp án:**
- Lặp 30 lần. Mỗi lần che 12% giá trị **đã quan sát được** trong file 03 rồi điền lại và đo sai số.
- Riêng income được che theo đúng tỷ lệ thiếu MAR ước lượng từ 03: 32% / 11% / 5% theo nhóm tuổi.
- Vì ba cách dùng chung mặt nạ trong mỗi lần lặp, có thể so sánh theo cặp.
- **Quy tắc:** chỉ dùng cách phức tạp nếu nó tốt hơn trung vị ít nhất 20/30 lần.

| Cột | Được chọn | Số lần thắng trung vị (03) | NMAE trung vị → chọn, so với file 01 |
|---|---|---|---|
| annual_income | KNN | 30/30 | 0.508 → **0.271** (MAE 95.2 → 50.8 triệu/năm) |
| avg_session_minutes | KNN | 30/30 | 0.525 → **0.227** (MAE 20.0 → 8.7 phút) |
| city_tier | trung vị (theo quy tắc) | 10/30 (iterative) | độ chính xác 0.30; KNN đạt 0.40 |
| age, tenure, recency, return_rate (chỉ có ô lỗi) | iterative, KNN, iterative, KNN | 22-29/30 | xem E2 |

**Các điểm cần nêu trong báo cáo:**
- Kiểm định nội bộ trên file 03 xếp hạng **đúng** cách tốt nhất cho income và session, khớp với kết quả so với file 01. Điều này chứng minh có thể chọn cách điền mà không cần dữ liệu sạch.
- **Sai lệch hệ thống của cách trung vị** đúng như dự đoán ở Câu b. Với khách 18-27 tuổi, trung vị điền cao hơn thực tế trung bình **+39.4** triệu/năm, còn KNN chỉ lệch **-1.7** (hình `c_eval_imputation.png`).
- Với `city_tier`, không cách nào tốt hơn đoán ngẫu nhiên: lớp phổ biến nhất chiếm 38.5%, và độ chính xác ở kiểm định nội bộ chỉ khoảng 33-35%. Điều này phù hợp với kết quả Câu b, city_tier hầu như không tương quan với các thuộc tính khác (|ρ| ≤ 0.27). Quy tắc dự phòng chọn trung vị. Trên file 01, KNN đúng hơn đúng 1 ô trong 10 ô, nên không đủ cơ sở để đổi quyết định. Cũng không đổi quyết định sau khi đã nhìn đáp án, vì làm vậy là rò rỉ thông tin.
- Các ô lỗi được điền với sai số lớn ở hai trường hợp: C0019 có tuổi thật 23 nhưng điền 33, và C0116 có return_rate thật 0.48 nhưng điền 0.22. Nguyên nhân là giá trị thật của hai khách này không điển hình so với hàng xóm của họ. Đây là giới hạn của mọi cách điền dựa trên hàng xóm.

## Nhiễu: đánh đổi

- **AOV.** Đối soát với total/orders chỉ khi tỷ lệ lệch rõ, tức là nằm ngoài dải IQR.
  - Được: không sửa nhầm bản ghi nào (precision 1.00). Sai số AOV trung bình trên toàn bộ dữ liệu giảm từ 3.67% xuống 1.66%.
  - Mất: 38/60 ô nhiễu nhỏ (sai số trung bình khoảng 6.8%) không bị phát hiện, vì không phân biệt được với biến động thật trong dải ±5%.
  - Giả định: total đáng tin hơn AOV. Nếu thực tế ngược lại, nhiễu sẽ chuyển từ total sang AOV.
- **Session và discount.** Không có quan hệ để đối soát. Có thể làm trơn hoặc rời rạc hóa (binning), nhưng cách đó xóa cả tín hiệu thật trên 100% bản ghi chỉ để giảm nhiễu ở khoảng 30% bản ghi. Vì vậy mình chọn không sửa từng ô, mà để Câu d dùng chuẩn hóa bền. Câu e có thể so sánh độ nhạy của các thuật toán với nhiễu.
- **Giá trị dồn ở biên** (discount = 0 hoặc 1, recency = 365): giữ nguyên. Đây là giá trị hợp lệ, và không có căn cứ để khôi phục giá trị trước khi bị cắt biên.

## Những gì Câu c KHÔNG làm, để dành cho Câu d

Câu c không biến đổi log, không chuẩn hóa, không mã hóa thuộc tính phân loại, không chọn hay bỏ thuộc tính (kể cả `total_spend_12m`). File 04 vẫn ở thang gốc để dễ kiểm chứng. Cờ `annual_income_was_missing` được giữ lại để Câu d quyết định có dùng hay không.
