"""
Câu c - Tiền xử lý (Data Preprocessing).
ĐẦU VÀO DUY NHẤT: data/03_raw_final.csv. Không đọc file 01, injection_log hay true_segment
(việc so sánh với dữ liệu sạch nằm ở cau_c/evaluate.py, chạy sau).

Các bước (dựa trên vấn đề tìm ra ở Câu b):
  A1. Lỗi miền giá trị -> sửa theo quan hệ xác định nếu có (total = orders × AOV), còn lại -> NaN
  A2. avg_session_minutes = 1440 -> nghi ngờ lỗi đơn vị -> NaN
  A3. Nhiễu: đối soát AOV với total/orders, chỉ sửa các bản ghi lệch ngoài dải IQR (thang log)
  A4. Chọn cách điền thiếu bằng kiểm định chéo NỘI BỘ (che giá trị đã quan sát trong 03, điền lại, đo sai số)
  A5. Điền bằng 3 cách (trung vị / KNN / Iterative), giữ cách tốt nhất theo A4 cho từng cột
  A6. Lưu data/04_preprocessed.csv, data/04_change_log.csv, cau_c/imputed_variants.csv
  A7. Hình trước/sau (figures/c_*.png)
Khách mua sỉ (orders_12m cực đoan nhưng nhất quán) được giữ nguyên.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA, FIG, HERE = os.path.join(ROOT, "data"), os.path.join(ROOT, "figures"), os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIG, exist_ok=True)
SEED, DPI = 24022375, 200
BLUE, ORANGE, RED, GREY, GREEN = "#2a6f97", "#e07a5f", "#d62828", "#adb5bd", "#2a9d8f"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.width", 200)

raw = pd.read_csv(os.path.join(DATA, "03_raw_final.csv"))
df = raw.copy()
changes = []   # log từng ô thay đổi


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def log_change(idx, col, new, step, reason):
    for i in np.atleast_1d(idx):
        changes.append(dict(customer_id=df.at[i, "customer_id"], column=col, old=raw.at[i, col],
                            new=new if np.isscalar(new) else new[i], step=step, reason=reason))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[hình] figures/{name}")


# ============================================================================ A1. Lỗi miền giá trị
section("A1. LỖI MIỀN GIÁ TRỊ (quy tắc từ Câu b)")
ratio = df["total_spend_12m"] / (df["orders_12m"] * df["avg_order_value"])

# (a) total sai đơn vị: tỷ lệ ≈ 1000 -> total là ô sai; sửa xác định total = orders × AOV
m = (ratio > 2) & (df["avg_order_value"] > 0)
for i in df.index[m]:
    new = round(df.at[i, "orders_12m"] * df.at[i, "avg_order_value"], -3)
    print(f"  {df.at[i,'customer_id']}: total/(orders×AOV) = {ratio[i]:.1f} ≈ 10^{np.log10(ratio[i]):.2f} "
          f"-> lỗi đơn vị ở total; sửa total = orders × AOV = {new:,.0f}")
    df.at[i, "total_spend_12m"] = new
    log_change(i, "total_spend_12m", new, "A1", "sai đơn vị (tỷ lệ ~1000): total = orders × AOV")

# (b) AOV ≤ 0: |AOV| khớp total/orders? -> sửa xác định AOV = total / orders
m = df["avg_order_value"] <= 0
for i in df.index[m]:
    new = round(df.at[i, "total_spend_12m"] / df.at[i, "orders_12m"], -3)
    print(f"  {df.at[i,'customer_id']}: AOV = {df.at[i,'avg_order_value']:,.0f} ≤ 0; total/orders = {new:,.0f} "
          f"(|AOV| lệch {abs(abs(df.at[i,'avg_order_value']) / new - 1):.1%}) -> sửa AOV = total / orders")
    df.at[i, "avg_order_value"] = new
    log_change(i, "avg_order_value", new, "A1", "AOV âm: AOV = total / orders")

# (c) các vi phạm không có quan hệ xác định để khôi phục -> NaN, điền ở A5
RULES_NAN = {
    "age": ~df["age"].between(18, 100),
    "tenure_months": df["tenure_months"] < 0,
    "recency_days": ~df["recency_days"].between(0, 365),
    "return_rate": ~df["return_rate"].between(0, 1),
    "discount_usage_rate": ~df["discount_usage_rate"].between(0, 1),
}
cell_type = pd.DataFrame("", index=df.index, columns=df.columns)   # loại ô: missing / error / suspect
cell_type[df.isna()] = "missing"
for col, m in RULES_NAN.items():
    for i in df.index[m]:
        print(f"  {df.at[i,'customer_id']}: {col} = {df.at[i, col]:g} vi phạm miền -> NaN, điền lại ở A5")
        cell_type.at[i, col] = "error"
        df.at[i, col] = np.nan

# ============================================================================ A2. 1440 phút
section("A2. avg_session_minutes = 1440 (ngoại lai không vi phạm miền)")
s = df["avg_session_minutes"]
q1, q3 = s.quantile([.25, .75]); fence = q3 + 1.5 * (q3 - q1)
sus = s > fence
for i in df.index[sus]:
    o = df.at[i, "orders_12m"]
    peers = df.loc[df["orders_12m"].between(o - 3, o + 3) & ~sus, "avg_session_minutes"]
    print(f"  {df.at[i,'customer_id']}: {s[i]:g} phút/tuần, ngưỡng IQR trên = {fence:.1f} (gấp {s[i]/fence:.1f} lần)")
    print(f"    1440 = 24 × 60 = đúng số phút trong MỘT NGÀY -> nghi ghi nhầm đơn vị (phút/ngày) hoặc giá trị mặc định")
    print(f"    Khách này có {o} đơn, recency {df.at[i,'recency_days']} ngày; khách có {o-3}-{o+3} đơn: "
          f"session trung vị {peers.median():.1f}, max {peers.max():.1f} -> 1440 không nhất quán với hành vi")
    print(f"    Phương án: (1) giữ - méo std {s.std():.1f} vs {s[~sus].std():.1f}; (2) winsorize về {fence:.1f} - vẫn là "
          f"giá trị bịa, cực đại giả; (3) CHỌN: coi là lỗi nghi ngờ -> NaN, điền theo hàng xóm, ghi cờ trong log")
    cell_type.at[i, "avg_session_minutes"] = "suspect"
    df.at[i, "avg_session_minutes"] = np.nan

# ============================================================================ A3. Nhiễu - đối soát AOV
section("A3. NHIỄU: đối soát avg_order_value với total_spend_12m / orders_12m")
ratio = df["total_spend_12m"] / (df["orders_12m"] * df["avg_order_value"])
lr = np.log(ratio)
q1, q3 = lr.quantile([.25, .75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
noisy = (lr < lo) | (lr > hi)
print(f"  log(total/(orders×AOV)): dải IQR [{np.exp(lo):.3f}, {np.exp(hi):.3f}] (tính trên 03 sau A1)")
print(f"  {noisy.sum()} bản ghi lệch ngoài dải -> AOV = total / orders (AOV theo định nghĩa là trung bình giá trị đơn)")
print(f"  Bản ghi trong dải ({(~noisy).sum()}): giữ nguyên - không sửa sai lệch nhỏ vì không phân biệt được với làm tròn/biến động thật")
print("  Đánh đổi: giả định total (tổng giao dịch) đáng tin hơn AOV; nếu total mới là ô nhiễu thì nhiễu chuyển sang AOV.")
print("  Với avg_session_minutes, discount_usage_rate: KHÔNG có quan hệ xác định để đối soát -> không sửa từng ô")
print("  (làm trơn/rời rạc hóa sẽ xóa cả tín hiệu thật; để Câu d dùng chuẩn hóa bền và thuật toán ít nhạy nhiễu).")
new_aov = (df["total_spend_12m"] / df["orders_12m"]).round(-3)
for i in df.index[noisy]:
    log_change(i, "avg_order_value", new_aov[i], "A3", f"đối soát nhiễu: tỷ lệ {ratio[i]:.3f} ngoài dải")
    df.at[i, "avg_order_value"] = new_aov[i]
ratio_after = df["total_spend_12m"] / (df["orders_12m"] * df["avg_order_value"])

# ============================================================================ A4. Kiểm định chéo nội bộ
section("A4. CHỌN CÁCH ĐIỀN THIẾU BẰNG KIỂM ĐỊNH CHÉO NỘI BỘ (chỉ dùng dữ liệu đã quan sát trong 03)")
IMP = ["age", "tenure_months", "annual_income", "recency_days", "return_rate", "avg_session_minutes", "city_tier"]
FEAT = IMP + ["orders_12m", "avg_order_value", "discount_usage_rate"]   # bỏ total: dư thừa (Câu b, R² = 0.997)
LOG10 = ["annual_income", "orders_12m", "avg_order_value"]
LOG1P = ["recency_days", "avg_session_minutes"]
INT_COLS = {"age": (18, 100), "tenure_months": (0, None), "recency_days": (0, 365), "city_tier": (1, 3)}
FLOAT_BOUNDS = {"return_rate": (0, 1), "avg_session_minutes": (0, 10080), "annual_income": (0, None)}


def fwd(X):
    Z = X.copy()
    Z[LOG10] = np.log10(Z[LOG10]); Z[LOG1P] = np.log1p(Z[LOG1P])
    mu, sd = Z.mean(), Z.std()
    return (Z - mu) / sd, (mu, sd)


def inv(Z, par):
    mu, sd = par
    X = Z * sd + mu
    X[LOG10] = 10 ** X[LOG10]; X[LOG1P] = np.expm1(X[LOG1P])
    for c, (a, b) in INT_COLS.items():
        X[c] = X[c].round().clip(a, b)
    for c, (a, b) in FLOAT_BOUNDS.items():
        X[c] = X[c].clip(a, b)
    return X


def impute(X, method):
    """X: DataFrame FEAT có NaN. Trả về DataFrame đã điền (thang gốc)."""
    if method == "median":
        out = X.fillna(X.median())
        for c, (a, b) in INT_COLS.items():
            out[c] = out[c].round()
        return out
    Z, par = fwd(X)
    if method == "knn":
        imp = KNNImputer(n_neighbors=5, weights="distance")
    else:
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=25, random_state=SEED, tol=1e-4)
    Zf = pd.DataFrame(imp.fit_transform(Z), index=Z.index, columns=Z.columns)
    out = inv(Zf, par)
    return X.where(X.notna(), out)          # chỉ thay ô NaN, giữ nguyên ô quan sát được


METHODS = ["median", "knn", "iterative"]
X0 = df[FEAT].copy()
rng = np.random.default_rng(SEED)
# Mô phỏng cơ chế thiếu như quan sát ở Câu b: income thiếu MAR theo nhóm tuổi (tỷ lệ ước lượng từ 03), cột khác MCAR
age_grp = pd.cut(X0["age"], [17, 27, 39, 100], labels=False)
miss_rate_by_age = raw["annual_income"].isna().groupby(pd.cut(raw["age"].where(raw["age"].between(18, 100)),
                                                              [17, 27, 39, 100], labels=False)).mean()
REPS, FRAC = 30, 0.12
res = []
for r in range(REPS):
    Xm = X0.copy(); masks = {}
    for c in IMP:
        obs = X0.index[X0[c].notna()]
        if c == "annual_income":
            w = age_grp[obs].map(miss_rate_by_age).fillna(miss_rate_by_age.mean()).values; w = w / w.sum()
        else:
            w = None
        pick = rng.choice(obs, size=int(FRAC * len(obs)), replace=False, p=w)
        masks[c] = pick; Xm.loc[pick, c] = np.nan
    for meth in METHODS:
        Xi = impute(Xm, meth)
        for c in IMP:
            t, p = X0.loc[masks[c], c].values, Xi.loc[masks[c], c].values
            iqr = np.subtract(*X0[c].quantile([.75, .25]))
            res.append(dict(rep=r, column=c, method=meth, MAE=np.abs(t - p).mean(), NMAE=np.abs(t - p).mean() / iqr,
                            acc=(np.round(p) == t).mean() if c == "city_tier" else np.nan))
cv = pd.DataFrame(res)
summ = cv.groupby(["column", "method"]).agg(MAE=("MAE", "mean"), NMAE=("NMAE", "mean"), NMAE_sd=("NMAE", "std"),
                                            acc=("acc", "mean")).reset_index()
tbl = summ.pivot(index="column", columns="method", values="NMAE")[METHODS].reindex(IMP)
print(f"{REPS} lần lặp, mỗi lần che {FRAC:.0%} giá trị quan sát được của mỗi cột (income: che theo tỷ lệ thiếu MAR "
      f"ước lượng từ 03 theo tuổi {miss_rate_by_age.round(3).to_dict()})")
print("\nSai số tuyệt đối trung bình chuẩn hóa theo IQR (NMAE, càng nhỏ càng tốt):")
print(tbl.round(3).to_string())
print("\nMAE trên thang gốc:")
print(summ.pivot(index="column", columns="method", values="MAE")[METHODS].reindex(IMP).round(3).to_string())
print("\ncity_tier - độ chính xác:", summ[summ.column == "city_tier"].set_index("method")["acc"].round(3).to_dict())
CHOSEN = tbl.idxmin(axis=1).to_dict()
print("\nCách được chọn cho từng cột (NMAE nhỏ nhất):", CHOSEN)
# So sánh CẶP (cùng mặt nạ trong mỗi lần lặp): độ lệch chuẩn ở trên phần lớn do mặt nạ khác nhau giữa các lần,
# nên chồng lấn thanh sai số không có nghĩa là không phân biệt được.
wide = cv.pivot_table(index=["rep", "column"], columns="method", values="NMAE").reset_index()
pw = []
for c in IMP:
    w = wide[wide.column == c]; order = tbl.loc[c].sort_values().index.tolist()
    best, second = order[0], order[1]
    diff = w[second] - w[best]
    pw.append(dict(column=c, chọn=best, thứ_hai=second, thắng=f"{(diff > 0).sum()}/{len(diff)}",
                   chênh_NMAE=round(diff.mean(), 3), so_với_median=f"{(w['median'] > w[best]).sum()}/{len(diff)}"))
pw = pd.DataFrame(pw)
print("\nSo sánh cặp trên cùng mặt nạ (số lần cách được chọn tốt hơn):")
print(pw.to_string(index=False))
print("Đọc: 'thắng' gần 15/30 = không phân biệt được với cách thứ hai; 'so_với_median' cho biết lợi ích so với cách đơn giản nhất.")
# Quy tắc chọn: cách phức tạp chỉ được dùng nếu tốt hơn trung vị một cách ổn định (≥ 2/3 số lần lặp)
for r in pw.itertuples():
    if int(r.so_với_median.split("/")[0]) < 20 and r.chọn != "median":
        print(f"  {r.column}: {r.chọn} chỉ tốt hơn median {r.so_với_median} lần -> không ổn định, dùng median (đơn giản hơn)")
        CHOSEN[r.column] = "median"
print("Cách dùng cuối cùng:", CHOSEN)

# ============================================================================ A5. Điền
section("A5. ĐIỀN GIÁ TRỊ THIẾU / LỖI")
print("Số ô cần điền theo loại:")
need = df[IMP].isna()
print(pd.DataFrame({t: [(need[c] & (cell_type[c].replace('', 'missing') == t)).sum() for c in IMP]
                    for t in ["missing", "error", "suspect"]}, index=IMP).to_string())
full = {m: impute(df[FEAT], m) for m in METHODS}
variants = []
for c in IMP:
    for i in df.index[need[c]]:
        variants.append(dict(customer_id=df.at[i, "customer_id"], column=c,
                             cell_type=cell_type.at[i, c] or "missing", raw_value=raw.at[i, c],
                             **{m: full[m].at[i, c] for m in METHODS}, chosen_method=CHOSEN[c],
                             chosen=full[CHOSEN[c]].at[i, c]))
variants = pd.DataFrame(variants)
income_missing = raw["annual_income"].isna().astype(int)
for c in IMP:
    idx = df.index[need[c]]
    vals = full[CHOSEN[c]][c]
    log_change(idx, c, vals, "A5", f"điền bằng {CHOSEN[c]}")
    df.loc[idx, c] = vals[idx]
for c in ["age", "tenure_months", "recency_days", "city_tier"]:
    df[c] = df[c].round().astype(int)
df["annual_income_was_missing"] = income_missing
assert df.drop(columns="annual_income_was_missing").isna().sum().sum() == 0

# ============================================================================ A6. Lưu
section("A6. LƯU KẾT QUẢ")
df.to_csv(os.path.join(DATA, "04_preprocessed.csv"), index=False)
ch = pd.DataFrame(changes)
ch["step_type"] = ch.apply(lambda r: "A5-" + (cell_type.loc[raw.index[raw.customer_id == r.customer_id][0], r.column] or "missing")
                           if r.step == "A5" else r.step, axis=1)
ch.to_csv(os.path.join(DATA, "04_change_log.csv"), index=False)
variants.to_csv(os.path.join(HERE, "imputed_variants.csv"), index=False)
summ.to_csv(os.path.join(HERE, "internal_cv.csv"), index=False)
print(f"data/04_preprocessed.csv: {df.shape[0]} bản ghi × {df.shape[1]} cột (thêm cờ annual_income_was_missing)")
print(f"data/04_change_log.csv: {len(ch)} ô thay đổi")
print(ch.groupby(["step_type", "column"]).size().to_string())

# Kiểm tra sau xử lý
viol = {"age": ~df["age"].between(18, 100), "tenure": df["tenure_months"] < 0, "AOV": df["avg_order_value"] <= 0,
        "recency": ~df["recency_days"].between(0, 365), "return": ~df["return_rate"].between(0, 1),
        "discount": ~df["discount_usage_rate"].between(0, 1),
        "ratio": ~(df["total_spend_12m"] / (df["orders_12m"] * df["avg_order_value"])).between(0.5, 2)}
print("Vi phạm miền còn lại:", {k: int(v.sum()) for k, v in viol.items()})
q = df["orders_12m"].quantile([.25, .75]); ws = df["orders_12m"] > q.iloc[1] + 1.5 * (q.iloc[1] - q.iloc[0])
print(f"Khách số đơn cực đoan (orders > {q.iloc[1] + 1.5*(q.iloc[1]-q.iloc[0]):g}): {ws.sum()} bản ghi; "
      f"số ô thay đổi = {int(ch.customer_id.isin(df.loc[ws, 'customer_id']).sum())} "
      f"(chỉ được phép là ô thiếu được điền)")

# ============================================================================ A7. Hình
section("A7. HÌNH TRƯỚC / SAU")
# (1) ô lỗi / nghi ngờ: trước -> sau
fix = ch[ch.step_type.isin(["A1", "A5-error", "A5-suspect"])]
cols = fix["column"].unique().tolist()
fig, axes = plt.subplots(1, len(cols), figsize=(2.9 * len(cols), 4.4))
for ax, c in zip(np.atleast_1d(axes), cols):
    rows = fix[fix.column == c]
    ax.scatter(np.zeros(len(df)) + np.random.default_rng(0).uniform(-.12, .12, len(df)), df[c], s=6, color=GREY, alpha=.5)
    for r in rows.itertuples():
        ax.plot([-.35, .35], [r.old, r.new], color=RED, lw=1)
        ax.scatter([-.35], [r.old], marker="X", color=RED, s=50, zorder=3)
        ax.scatter([.35], [r.new], marker="o", color=GREEN, s=40, zorder=3)
        ax.annotate(f"{r.old:g}", (-.35, r.old), xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=7)
        ax.annotate(f"{r.new:g}", (.35, r.new), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xticks([-.35, .35], ["03 (trước)", "04 (sau)"]); ax.set_xlim(-.8, .8)
    ax.set_title(c, fontsize=9)
    if c in ("total_spend_12m",):
        ax.set_yscale("log")
fig.suptitle("Ô lỗi / nghi ngờ: giá trị trước (X đỏ) → sau (chấm xanh); nền xám = toàn bộ giá trị trong 04", fontsize=11)
fig.tight_layout()
save(fig, "c_errors_fixed.png")

# (2) kiểm định chéo nội bộ
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.27
for k, (m, col) in enumerate(zip(METHODS, [GREY, BLUE, ORANGE])):
    d = summ[summ.method == m].set_index("column").reindex(IMP)
    ax.bar(np.arange(len(IMP)) + (k - 1) * w, d["NMAE"], w, yerr=d["NMAE_sd"], capsize=2, color=col, label=m)
ax.set_xticks(range(len(IMP)), [f"{c}\n→ {CHOSEN[c]}" for c in IMP], fontsize=8.5)
ax.set_ylabel("NMAE = MAE / IQR (che 12%, 30 lần)")
ax.set_title("Kiểm định chéo nội bộ trên file 03: sai số điền lại giá trị đã biết (± độ lệch chuẩn qua 30 lần)")
ax.legend()
fig.tight_layout()
save(fig, "c_imputation_cv.png")

# (3) income: 3 cách điền theo tuổi
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
obs = raw["annual_income"].notna() & raw["age"].between(18, 100)
mi = variants[variants.column == "annual_income"].merge(df[["customer_id", "age"]], on="customer_id")
for ax, m in zip(axes, METHODS):
    ax.scatter(raw.loc[obs, "age"], raw.loc[obs, "annual_income"], s=12, color=GREY, alpha=.6, label="quan sát được")
    ax.scatter(mi["age"], mi[m], s=36, color=ORANGE, edgecolor="black", lw=.4, label=f"điền ({len(mi)})")
    ax.set_yscale("log"); ax.set_xlabel("age")
    ax.set_title(f"{m}" + ("  (được chọn)" if CHOSEN["annual_income"] == m else ""))
axes[0].set_ylabel("annual_income (triệu/năm, log)"); axes[0].legend(fontsize=8)
fig.suptitle("annual_income thiếu theo tuổi (MAR): trung vị gán mọi khách trẻ cùng một mức; KNN/Iterative theo xu hướng tuổi - thu nhập",
             fontsize=11)
fig.tight_layout()
save(fig, "c_income_imputation.png")

# (4) đối soát AOV
fig, ax = plt.subplots(figsize=(8, 3.8))
bins = np.linspace(0.7, 1.35, 40)
ax.hist(ratio.clip(.7, 1.35), bins=bins, color=ORANGE, alpha=.6, label="sau A1, trước A3")
ax.hist(ratio_after.clip(.7, 1.35), bins=bins, color=BLUE, alpha=.6, label="sau A3")
ax.axvspan(np.exp(lo), np.exp(hi), color="0.9", zorder=-1, label=f"dải IQR [{np.exp(lo):.3f}, {np.exp(hi):.3f}]")
ax.set_xlabel("total_spend_12m / (orders_12m × avg_order_value)  (cắt hiển thị ở [0.7, 1.35])")
ax.set_ylabel("Số bản ghi"); ax.legend(fontsize=8)
ax.set_title(f"Đối soát nhiễu AOV: {noisy.sum()} bản ghi ngoài dải được tính lại AOV = total / orders")
fig.tight_layout()
save(fig, "c_aov_reconcile.png")

# (5) phân bố trước / sau cho các cột được điền
fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, c in zip(axes, ["annual_income", "avg_session_minutes", "age", "recency_days"]):
    b = raw[c].copy()
    if c in RULES_NAN: b = b.where(RULES_NAN[c].pipe(lambda m: ~m))
    if c == "avg_session_minutes": b = b.where(~sus)
    b = b.dropna(); a = df[c]
    xb, xa = (np.log10(b), np.log10(a)) if c == "annual_income" else (b, a)
    edges = np.histogram_bin_edges(np.concatenate([xb, xa]), 25)
    ax.hist(xb, edges, color=BLUE, alpha=.55, label=f"03 quan sát hợp lệ (n={len(b)})")
    ax.hist(xa, edges, color=ORANGE, alpha=.55, label=f"04 sau điền (n={len(a)})")
    ax.set_title(f"{c}: median {b.median():.3g} → {a.median():.3g}", fontsize=10)
    ax.set_xlabel("log10(annual_income)" if c == "annual_income" else c)
    ax.legend(fontsize=7.5)
fig.suptitle("Phân bố trước (giá trị hợp lệ quan sát được trong 03) và sau tiền xử lý (04)", fontsize=11)
fig.tight_layout()
save(fig, "c_distributions_before_after.png")
print("\nXong preprocess.py. Chạy tiếp cau_c/evaluate.py để so với dữ liệu sạch.")
