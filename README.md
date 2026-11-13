# DataMining_Midterm

Đề thi giữa kỳ môn Phân tích và khai phá dữ liệu (UET, HK I 2026-2027) - bài toán phân cụm khách hàng sàn TMĐT.

## Tham số theo MSSV

s = 24022375 → a = s mod 5 = 0, b = (s div 10) mod 5 = 2, c = (s div 100) mod 5 = 3

| Đặc điểm | Yêu cầu | Thực hiện |
|---|---|---|
| Số bản ghi | N = 200 + 20a = 200 | 200 |
| Thuộc tính đầu vào | ≥ 8 + b = 10 | 14 |
| Giá trị thiếu | ≥ 2 thuộc tính | 3 (`annual_income` MAR theo tuổi, `avg_session_minutes` MCAR, `city_tier` MCAR) |
| Nhiễu ngẫu nhiên | ≥ 2 thuộc tính số | 3 (`avg_order_value`, `avg_session_minutes`, `discount_usage_rate`) |
| Ngoại lai | ≥ 5 + 2c = 11 bản ghi | 14 (8 lỗi nhập liệu + 6 khách mua sỉ hợp lệ) |

## Cấu trúc

```
cau_a/generate_data.py          # sinh dữ liệu, seed = MSSV, tự kiểm tra ràng buộc
data/
  01_clean_ground_truth.csv     # dữ liệu sạch + nhãn phân khúc ẩn - CHỈ dùng đối chiếu
  02_noisy_outliers.csv         # sau khi thêm nhiễu + ngoại lai
  03_raw_final.csv              # sau khi thêm giá trị thiếu - đầu vào cho câu b-e
  injection_log.csv             # log từng ô bị tác động (giá trị cũ / mới)
  data_dictionary.csv           # mô tả thuộc tính
```

## Tái lập

```bash
pip install numpy pandas
python cau_a/generate_data.py
```

Chạy lại sẽ sinh ra đúng các file trong `data/` (seed cố định = 24022375).

## Thiết kế sinh dữ liệu (Câu a)

- 4 phân khúc ẩn: VIP (40), Bargain (60), Occasional (60), ChurnRisk (40).
- Phân phối: tuổi/tenure/phút truy cập ~ Normal (cắt biên); thu nhập, giá trị đơn ~ Log-normal; số đơn ~ Poisson; recency ~ Exponential; tỷ lệ dùng mã giảm giá, tỷ lệ trả hàng ~ Beta.
- `total_spend_12m` ≈ `orders_12m` × `avg_order_value` - cố ý dư thừa để phân tích tương quan và chọn thuộc tính.
- Ngoại lai lỗi: tuổi 150 và 3, tenure âm, AOV âm, recency 9999, return_rate 1.8, 1440 phút/tuần, tổng chi tiêu sai đơn vị (×1000).
- Ngoại lai hợp lệ: khách mua sỉ - số đơn 120-260, AOV ~3.5 triệu, các thuộc tính nhất quán nội tại.

**Hạn chế:** thuộc tính gần như độc lập có điều kiện trong mỗi phân khúc; ranh giới phân khúc rõ hơn dữ liệu thực.

**Nguyên tắc:** file `01_clean_ground_truth.csv` và cột `true_segment` không được dùng để tiền xử lý; chỉ dùng để đánh giá (ARI, sai số imputation).
