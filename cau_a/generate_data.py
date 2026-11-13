"""
Câu a - Thiết kế và sinh dữ liệu khách hàng TMĐT cho bài toán phân cụm
MSSV s = 24022375 -> a = 0, b = 2, c = 3
  N = 200 + 20a = 200 bản ghi
  Thuộc tính đầu vào >= 8 + b = 10   (thiết kế: 14)
  Thiếu >= 2 thuộc tính             (thiết kế: 3 - annual_income MAR, avg_session_minutes MCAR, city_tier MCAR)
  Nhiễu >= 2 thuộc tính số          (thiết kế: 3 - avg_order_value, avg_session_minutes, discount_usage_rate)
  Ngoại lai >= 5 + 2c = 11 bản ghi  (thiết kế: 14 = 8 lỗi nhập liệu + 6 ngoại lai hợp lệ)

Đầu ra (thư mục data/):
  01_clean_ground_truth.csv    - dữ liệu sạch + nhãn phân khúc ẩn (CHỈ dùng đối chiếu)
  02_noisy_outliers.csv        - sau khi thêm nhiễu + ngoại lai
  03_raw_final.csv             - sau khi thêm giá trị thiếu (dữ liệu đầu vào cho câu b-e, KHÔNG có nhãn)
  injection_log.csv            - log từng ô bị tác động (nhiễu / ngoại lai / thiếu)
  data_dictionary.csv          - mô tả thuộc tính
"""
import os
import numpy as np
import pandas as pd

SEED = 24022375
rng = np.random.default_rng(SEED)
N = 200
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. Phân khúc ẩn
SEGMENTS = {
    # tên: (tỷ lệ, tham số)
    "VIP":          0.20,
    "Bargain":      0.30,
    "Occasional":   0.30,
    "ChurnRisk":    0.20,
}
names = list(SEGMENTS)
counts = (np.array(list(SEGMENTS.values())) * N).astype(int)
segment = np.repeat(names, counts)
rng.shuffle(segment)

P = {  # tham số sinh theo phân khúc
    #             age(mu,sd) tenure(mu,sd) income_ln(mu,sd) orders(lam) aov_ln(mu,sd) recency(scale) disc(a,b) ret(a,b) sess(mu,sd)
    "VIP":        ((38, 7),  (48, 12), (np.log(450), 0.30), 28, (np.log(1.6e6), 0.30), 10,  (2, 8), (2, 30), (45, 10)),
    "Bargain":    ((29, 6),  (24, 10), (np.log(180), 0.30), 18, (np.log(2.5e5), 0.30), 15,  (8, 2), (3, 20), (60, 12)),
    "Occasional": ((33, 9),  (12, 8),  (np.log(250), 0.35), 4,  (np.log(5.0e5), 0.35), 60,  (3, 5), (2, 25), (20, 6)),
    "ChurnRisk":  ((35, 10), (30, 14), (np.log(220), 0.35), 2,  (np.log(4.0e5), 0.35), 200, (3, 4), (6, 12), (10, 4)),
}
CAT_PREF = {
    "VIP":        [0.30, 0.10, 0.35, 0.10, 0.15],
    "Bargain":    [0.15, 0.40, 0.10, 0.25, 0.10],
    "Occasional": [0.25, 0.20, 0.20, 0.20, 0.15],
    "ChurnRisk":  [0.20, 0.25, 0.15, 0.25, 0.15],
}
CATEGORIES = ["Electronics", "Fashion", "Home_Living", "Grocery", "Beauty"]
PAY_PREF = {
    "VIP":        [0.55, 0.30, 0.10, 0.05],
    "Bargain":    [0.15, 0.45, 0.20, 0.20],
    "Occasional": [0.25, 0.30, 0.30, 0.15],
    "ChurnRisk":  [0.15, 0.25, 0.50, 0.10],
}
PAYMENTS = ["Credit_Card", "E_Wallet", "COD", "Bank_Transfer"]
TIER_PREF = {  # city_tier 1 (TP lớn) - 3 (tỉnh)
    "VIP": [0.65, 0.25, 0.10], "Bargain": [0.40, 0.35, 0.25],
    "Occasional": [0.35, 0.35, 0.30], "ChurnRisk": [0.30, 0.35, 0.35],
}

rows = []
for i, seg in enumerate(segment):
    (am, asd), (tm, tsd), (im, isd), lam, (vm, vsd), rsc, (da, db), (ra, rb), (sm, ssd) = P[seg]
    age = int(np.clip(rng.normal(am, asd), 18, 70))
    tenure = int(np.clip(rng.normal(tm, tsd), 1, 96))
    income = float(np.exp(rng.normal(im, isd)))                    # triệu VND/năm
    income *= 1 + 0.01 * (age - 30)                                 # thu nhập tăng nhẹ theo tuổi
    orders = int(max(1, rng.poisson(lam)))
    aov = float(np.exp(rng.normal(vm, vsd)))                        # VND
    total = orders * aov * rng.uniform(0.95, 1.05)                  # thuộc tính dẫn xuất (tương quan cao)
    recency = int(np.clip(rng.exponential(rsc), 0, 365))
    disc = float(rng.beta(da, db))
    ret = float(rng.beta(ra, rb))
    sess = float(max(1, rng.normal(sm, ssd)))
    rows.append(dict(
        customer_id=f"C{i+1:04d}",
        age=age,
        gender=rng.choice(["F", "M"], p=[0.55, 0.45]),
        city_tier=int(rng.choice([1, 2, 3], p=TIER_PREF[seg])),
        tenure_months=tenure,
        annual_income=round(income, 1),
        orders_12m=orders,
        avg_order_value=round(aov, -3),
        total_spend_12m=round(total, -3),
        recency_days=recency,
        discount_usage_rate=round(disc, 3),
        return_rate=round(ret, 3),
        avg_session_minutes=round(sess, 1),
        preferred_category=rng.choice(CATEGORIES, p=CAT_PREF[seg]),
        payment_method=rng.choice(PAYMENTS, p=PAY_PREF[seg]),
        true_segment=seg,
    ))
clean = pd.DataFrame(rows)
clean.to_csv(f"{OUT}/01_clean_ground_truth.csv", index=False)

log = []
df = clean.drop(columns=["true_segment"]).copy()
df["avg_order_value"] = df["avg_order_value"].astype(float)
df["total_spend_12m"] = df["total_spend_12m"].astype(float)
df["age"] = df["age"].astype(float)
df["tenure_months"] = df["tenure_months"].astype(float)

# Nguyên tắc: mỗi ô chỉ bị tác động TỐI ĐA 1 lần (nhiễu / ngoại lai / thiếu loại trừ nhau)
# để log khớp 1-1 với dữ liệu cuối. Chọn trước bản ghi ngoại lai, nhiễu chỉ áp lên bản ghi còn lại.
out_rows = rng.choice(N, size=14, replace=False)
err_rows, legit_rows = out_rows[:8], out_rows[8:]
non_out = np.setdiff1d(np.arange(N), out_rows)

# ---------------------------------------------------------------- 2. Nhiễu ngẫu nhiên (Gaussian, 60 bản ghi/thuộc tính, ngoài 14 bản ghi ngoại lai)
NOISE = {  # thuộc tính: (độ lệch chuẩn nhiễu tương đối hay tuyệt đối, kiểu)
    "avg_order_value":     ("mult", 0.15),   # nhân (1 + N(0, 0.15))
    "avg_session_minutes": ("add", 6.0),     # cộng N(0, 6)
    "discount_usage_rate": ("add", 0.08),    # cộng N(0, 0.08), cắt [0,1]
}
for col, (kind, sd) in NOISE.items():
    idx = rng.choice(non_out, size=int(0.30 * N), replace=False)
    for j in idx:
        old = df.at[j, col]
        # Rút lại nhiễu nếu sau làm tròn/cắt biên giá trị không đổi (tối đa 100 lần).
        # Hệ quả: phân phối nhiễu thực tế là N(0, sd) bỏ đi vùng |nhiễu| < 1/2 bước làm tròn
        # (1000 VND, 0.1 phút, 0.001) - sai lệch không đáng kể so với sd.
        for _ in range(100):
            if kind == "mult":
                new = round(old * (1 + rng.normal(0, sd)), -3)
            else:
                new = old + rng.normal(0, sd)
            if col == "discount_usage_rate":
                new = round(float(np.clip(new, 0, 1)), 3)
            elif col == "avg_session_minutes":
                new = round(max(0.5, new), 1)
            if not np.isclose(new, old):
                break
        else:
            raise RuntimeError(f"Không tạo được nhiễu khác giá trị gốc tại ({j}, {col})")
        df.at[j, col] = new
        log.append(dict(row=j, customer_id=df.at[j, "customer_id"], column=col,
                        type="noise", old=old, new=new))

# ---------------------------------------------------------------- 3. Ngoại lai (14 bản ghi đã chọn ở trên, không trùng nhau)

ERROR_SPECS = [  # lỗi nhập liệu / lỗi hệ thống -> nên sửa hoặc coi là thiếu
    ("age", lambda v: 150.0),
    ("age", lambda v: 3.0),
    ("tenure_months", lambda v: -5.0),
    ("avg_order_value", lambda v: -abs(v)),
    ("recency_days", lambda v: 9999),
    ("return_rate", lambda v: 1.8),
    ("avg_session_minutes", lambda v: 1440.0),
    ("total_spend_12m", lambda v: v * 1000),          # sai đơn vị
]
for j, (col, f) in zip(err_rows, ERROR_SPECS):
    old = df.at[j, col]
    new = f(old)
    df.at[j, col] = new
    log.append(dict(row=j, customer_id=df.at[j, "customer_id"], column=col,
                    type="outlier_error", old=old, new=new))

# Ngoại lai hợp lệ: khách mua sỉ / doanh nghiệp nhỏ - giá trị cực lớn nhưng NHẤT QUÁN nội tại
for j in legit_rows:
    changes = {
        "orders_12m": int(rng.integers(120, 260)),
        "avg_order_value": float(round(np.exp(rng.normal(np.log(3.5e6), 0.3)), -3)),
        "annual_income": round(float(rng.uniform(1500, 4000)), 1),
        "discount_usage_rate": round(float(rng.uniform(0.3, 0.6)), 3),
        "recency_days": int(rng.integers(0, 5)),
    }
    changes["total_spend_12m"] = float(round(changes["orders_12m"] * changes["avg_order_value"], -3))
    for col, new in changes.items():
        old = df.at[j, col]
        df.at[j, col] = new
        log.append(dict(row=j, customer_id=df.at[j, "customer_id"], column=col,
                        type="outlier_legit", old=old, new=new))

df.to_csv(f"{OUT}/02_noisy_outliers.csv", index=False)

# ---------------------------------------------------------------- 4. Giá trị thiếu
# Chỉ xóa những ô CHƯA bị nhiễu/ngoại lai (tránh xóa mất ô đã ghi log).
# Hệ quả (ghi vào báo cáo): cơ chế MCAR/MAR áp trên tập ô đủ điều kiện, không phải toàn bộ 200 dòng.
touched = set(zip(pd.DataFrame(log)["row"], pd.DataFrame(log)["column"]))
def eligible(col):
    return np.array([j for j in range(N) if (j, col) not in touched])

# 4.1 annual_income - MAR: xác suất thiếu phụ thuộc age (khách trẻ ít khai thu nhập)
age_c = clean["age"].values
p_miss = np.where(age_c < 28, 0.35, np.where(age_c < 40, 0.12, 0.05))
el = eligible("annual_income")
miss_income = el[rng.random(len(el)) < p_miss[el]]
# 4.2 avg_session_minutes - MCAR 8% (16 ô)
miss_sess = rng.choice(eligible("avg_session_minutes"), size=int(0.08 * N), replace=False)
# 4.3 city_tier - MCAR 5% (10 ô)
miss_tier = rng.choice(eligible("city_tier"), size=int(0.05 * N), replace=False)

final = df.copy()
final["city_tier"] = final["city_tier"].astype("Int64")
for col, idx, mech in [("annual_income", miss_income, "MAR(age)"),
                       ("avg_session_minutes", miss_sess, "MCAR"),
                       ("city_tier", miss_tier, "MCAR")]:
    for j in idx:
        log.append(dict(row=j, customer_id=final.at[j, "customer_id"], column=col,
                        type=f"missing_{mech}", old=final.at[j, col], new=np.nan))
        final.at[j, col] = pd.NA if col == "city_tier" else np.nan

final.to_csv(f"{OUT}/03_raw_final.csv", index=False)
pd.DataFrame(log).to_csv(f"{OUT}/injection_log.csv", index=False)

# ---------------------------------------------------------------- 5. Data dictionary
dd = [
    ("customer_id", "ID", "string", "Mã khách hàng - không dùng để phân cụm"),
    ("age", "Nhân khẩu học", "int", "Tuổi (18-70)"),
    ("gender", "Nhân khẩu học", "nominal", "F/M"),
    ("city_tier", "Nhân khẩu học", "ordinal", "1 = TP lớn, 2 = TP vừa, 3 = tỉnh"),
    ("tenure_months", "Quan hệ", "int", "Số tháng từ khi đăng ký"),
    ("annual_income", "Kinh tế", "float", "Thu nhập tự khai (triệu VND/năm)"),
    ("orders_12m", "Hành vi (F)", "int", "Số đơn hàng 12 tháng"),
    ("avg_order_value", "Hành vi (M)", "float", "Giá trị đơn trung bình (VND)"),
    ("total_spend_12m", "Hành vi (M)", "float", "Tổng chi tiêu 12 tháng (VND) - dẫn xuất ~ orders x AOV"),
    ("recency_days", "Hành vi (R)", "int", "Số ngày từ lần mua gần nhất"),
    ("discount_usage_rate", "Hành vi", "float [0,1]", "Tỷ lệ đơn dùng mã giảm giá"),
    ("return_rate", "Hành vi", "float [0,1]", "Tỷ lệ đơn bị trả hàng"),
    ("avg_session_minutes", "Tương tác", "float", "Phút truy cập trung bình/tuần"),
    ("preferred_category", "Sở thích", "nominal", "Ngành hàng mua nhiều nhất (5 giá trị)"),
    ("payment_method", "Sở thích", "nominal", "Phương thức thanh toán chính (4 giá trị)"),
]
pd.DataFrame(dd, columns=["attribute", "group", "dtype", "description"]).to_csv(
    f"{OUT}/data_dictionary.csv", index=False)

# ---------------------------------------------------------------- 6. Kiểm tra ràng buộc đề bài TRÊN FILE CUỐI (đọc lại từ đĩa)
lg = pd.read_csv(f"{OUT}/injection_log.csv")
f03 = pd.read_csv(f"{OUT}/03_raw_final.csv")
f01 = pd.read_csv(f"{OUT}/01_clean_ground_truth.csv")
COLS = [c for c in f03.columns if c != "customer_id"]

def same(a, b):
    if pd.isna(a) or pd.isna(b):
        return pd.isna(a) and pd.isna(b)
    try:
        return np.isclose(float(a), float(b))
    except (TypeError, ValueError):
        return str(a) == str(b)

# (i) mỗi ô bị tác động đúng 1 lần
assert not lg.duplicated(["row", "column"]).any(), "Có ô bị tác động nhiều lần"
# (ii) giá trị trong file 03 khớp cột 'new' của log
mismatch = [(r.row, r.column) for r in lg.itertuples() if not same(f03.at[r.row, r.column], r.new)]
assert not mismatch, f"Log không khớp file 03: {mismatch}"
# (iv) mọi tác động phải có hiệu lực: new khác old (và old phải bằng giá trị sạch)
no_effect = [(r.row, r.column, r.type) for r in lg.itertuples() if same(r.old, r.new)]
assert not no_effect, f"Tác động không làm đổi giá trị: {no_effect}"
wrong_old = [(r.row, r.column) for r in lg.itertuples() if not same(r.old, f01.at[r.row, r.column])]
assert not wrong_old, f"Cột 'old' không bằng giá trị sạch: {wrong_old}"
# (iii) mọi ô KHÔNG có trong log phải giống hệt dữ liệu sạch
logged = set(zip(lg.row, lg.column))
untouched_diff = [(j, c) for j in range(N) for c in COLS
                  if (j, c) not in logged and not same(f03.at[j, c], f01.at[j, c])]
assert not untouched_diff, f"Ô thay đổi ngoài log: {untouched_diff[:5]}"

n_inputs = len(COLS)
miss_cols = f03.columns[f03.isna().any()].tolist()
noise_cols = [c for c in lg.loc[lg.type == "noise", "column"].unique()
              if pd.api.types.is_numeric_dtype(f03[c])]
n_out = lg.loc[lg.type.str.startswith("outlier"), "row"].nunique()
print(f"N = {len(f03)}  (yêu cầu 200)")
print(f"Thuộc tính đầu vào = {n_inputs}  (yêu cầu >= 10)")
print(f"Thuộc tính có thiếu = {miss_cols}  (yêu cầu >= 2)")
print(f"Thuộc tính số có nhiễu (còn trong file 03) = {noise_cols}  (yêu cầu >= 2)")
print(f"Bản ghi ngoại lai (còn trong file 03) = {n_out}  (yêu cầu >= 11)")
print("Số ô thiếu:", f03.isna().sum()[f03.isna().sum() > 0].to_dict())
print("Số ô theo loại:", lg.type.value_counts().to_dict())
print("Phân khúc:", f01.true_segment.value_counts().to_dict())
assert len(f03) == 200 and n_inputs >= 10 and len(miss_cols) >= 2 and len(noise_cols) >= 2 and n_out >= 11
print("OK - log khớp 1-1 với file 03, mọi tác động có hiệu lực, mọi ràng buộc đạt.")
