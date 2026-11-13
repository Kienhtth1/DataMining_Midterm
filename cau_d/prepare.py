"""
Câu d - Data Preparation cho phân cụm.
Đầu vào: data/04_preprocessed.csv (và data/04_change_log.csv để dựng biến thể AOV gốc cho phân tích độ nhạy).
Không đọc file 01, injection_log hay true_segment.

  D1. Chọn thuộc tính: bỏ ID, bỏ total_spend_12m (dư thừa), bỏ cờ thiếu; giữ/bỏ thuộc tính phân loại theo kiểm định
  D2. Biến đổi: log cho thuộc tính lệch mạnh (|skew| > 1)
  D3. Mã hóa: city_tier giữ thứ bậc; định danh -> one-hot có trọng số cân bằng đóng góp
  D4. Chuẩn hóa: so sánh StandardScaler và RobustScaler, chọn theo ảnh hưởng của nhóm cực đoan
  D5. Giảm chiều: PCA - chọn số thành phần, đo mức bảo toàn khoảng cách (phương án thay thế cho Câu e)
  D6. Giảm số đối tượng: đánh giá có cần không
  D7. Phân tích thay đổi: đóng góp vào khoảng cách, tương quan, xu hướng cụm (Hopkins) trước/sau

Đầu ra: data/05_prepared.csv (ma trận chính), data/05_prepared_pca.csv, data/05_prepared_aov03.csv (độ nhạy A3),
        cau_d/prep_params.json, figures/d_*.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA, FIG, HERE = os.path.join(ROOT, "data"), os.path.join(ROOT, "figures"), os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIG, exist_ok=True)
SEED, DPI = 24022375, 200
BLUE, ORANGE, GREY, RED = "#2a6f97", "#e07a5f", "#adb5bd", "#d62828"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.width", 200)

df = pd.read_csv(os.path.join(DATA, "04_preprocessed.csv"))
NUM_ALL = ["age", "tenure_months", "annual_income", "orders_12m", "avg_order_value", "total_spend_12m",
           "recency_days", "discount_usage_rate", "return_rate", "avg_session_minutes"]
NOMINAL = ["gender", "preferred_category", "payment_method"]
ORDINAL = ["city_tier"]


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[hình] figures/{name}")


# ============================================================================ D1. Chọn thuộc tính
section("D1. CHỌN THUỘC TÍNH")
print("Bỏ customer_id: định danh, không mang thông tin hành vi.")
print("Bỏ total_spend_12m: Câu b - log(total) = log(orders) + log(AOV) với R² = 0.997, VIF = 388;")
print("   giữ log(orders) và log(AOV) giữ được cả 'mức chi tiêu' lẫn 'phong cách mua' (nhiều đơn rẻ / ít đơn đắt).")
L = np.log10(df[["orders_12m", "avg_order_value", "total_spend_12m"]])
A = np.column_stack([np.ones(len(L)), L[["orders_12m", "avg_order_value"]]])
res = L["total_spend_12m"] - A @ np.linalg.lstsq(A, L["total_spend_12m"], rcond=None)[0]
print(f"   Kiểm tra lại trên 04: R² = {1 - res.var() / L['total_spend_12m'].var():.4f}")
print("Bỏ annual_income_was_missing: là đặc điểm của quá trình thu thập dữ liệu, không phải hành vi;")
r = stats.pointbiserialr(df["annual_income_was_missing"], df["age"])
print(f"   lại tương quan với tuổi (r = {r[0]:.2f}, p = {r[1]:.4f}) -> đưa vào sẽ tạo trục phân cụm giả theo 'ai không khai thu nhập'.")

NUM = [c for c in NUM_ALL if c != "total_spend_12m"]
print("\nThuộc tính phân loại: giữ nếu liên quan ý nghĩa đến ≥ 1 thuộc tính số (Kruskal-Wallis, Bonferroni trên "
      f"{len(NUM)} phép thử mỗi thuộc tính, α = 0.05). Thuộc tính không liên quan hành vi chỉ thêm chiều nhiễu.")
assoc = []
for cat in NOMINAL + ORDINAL:
    groups = df[cat].unique()
    for c in NUM:
        h, p = stats.kruskal(*[df.loc[df[cat] == k, c] for k in groups])
        eps2 = (h - len(groups) + 1) / (len(df) - len(groups))       # epsilon² - cỡ hiệu ứng
        assoc.append(dict(cat=cat, num=c, p=p, p_bonf=min(1, p * len(NUM)), eps2=eps2))
assoc = pd.DataFrame(assoc)
summ = assoc.groupby("cat").agg(n_ý_nghĩa=("p_bonf", lambda s: int((s < .05).sum())), p_bonf_min=("p_bonf", "min"),
                                eps2_max=("eps2", "max")).reindex(NOMINAL + ORDINAL)
summ["giữ"] = summ["n_ý_nghĩa"] >= 1
print(summ.round(4).to_string())
KEEP_NOM = [c for c in NOMINAL if summ.at[c, "giữ"]]
KEEP_ORD = [c for c in ORDINAL if summ.at[c, "giữ"]]
print(f"-> Giữ định danh: {KEEP_NOM}; thứ bậc: {KEEP_ORD}; bỏ: {[c for c in NOMINAL + ORDINAL if not summ.at[c, 'giữ']]}")
cv = {}
for a in NOMINAL:
    for b in NOMINAL:
        if a < b:
            ct = pd.crosstab(df[a], df[b]); chi2 = stats.chi2_contingency(ct)[0]
            cv[f"{a}-{b}"] = round(float(np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))), 3)
print("Cramér's V giữa các thuộc tính định danh (không có cặp nào trùng lặp mạnh):", cv)

# ============================================================================ D2. Biến đổi
section("D2. BIẾN ĐỔI PHÂN BỐ (chỉ khi |skew| > 1; chọn biến đổi làm |skew| nhỏ nhất)")
TRANS = {}
rows = []
for c in NUM:
    x = df[c].astype(float)
    cands = {"none": x, "sqrt": np.sqrt(x), "log1p": np.log1p(x)}
    if (x > 0).all():
        cands["log10"] = np.log10(x)
    sk = {k: v.skew() for k, v in cands.items()}
    best = "none" if abs(sk["none"]) <= 1 else min(sk, key=lambda k: abs(sk[k]))
    TRANS[c] = best
    rows.append(dict(thuộc_tính=c, **{f"skew_{k}": sk.get(k, np.nan) for k in ["none", "sqrt", "log1p", "log10"]}, chọn=best))
print(pd.DataFrame(rows).set_index("thuộc_tính").round(2).to_string())


def transform(d):
    out = pd.DataFrame(index=d.index)
    for c in NUM:
        x = d[c].astype(float)
        f = {"none": lambda v: v, "sqrt": np.sqrt, "log1p": np.log1p, "log10": np.log10}[TRANS[c]]
        out[c] = f(x)
    for c in KEEP_ORD:
        out[c] = d[c].astype(float)
    return out


T = transform(df)

# ============================================================================ D4 (trước D3). Chuẩn hóa
section("D4. CHUẨN HÓA: StandardScaler (mean/std) hay RobustScaler (median/IQR)?")
q = df["orders_12m"].quantile([.25, .75]); fence = q.iloc[1] + 1.5 * (q.iloc[1] - q.iloc[0])
ext = df["orders_12m"] > fence
print(f"Nhóm cực đoan (orders_12m > {fence:g}, giữ lại theo Câu c): {ext.sum()} bản ghi")
sc = pd.DataFrame({"std (đủ)": T.std(), "std (bỏ cực đoan)": T[~ext].std(),
                   "IQR (đủ)": T.quantile(.75) - T.quantile(.25), "IQR (bỏ cực đoan)": T[~ext].quantile(.75) - T[~ext].quantile(.25)})
sc["Δstd %"] = (sc["std (đủ)"] / sc["std (bỏ cực đoan)"] - 1) * 100
sc["ΔIQR %"] = (sc["IQR (đủ)"] / sc["IQR (bỏ cực đoan)"] - 1) * 100
print(sc.round(3).to_string())
max_dstd = sc["Δstd %"].abs().max()
print(f"Sau biến đổi log, 6 bản ghi cực đoan làm std thay đổi tối đa {max_dstd:.1f}% ({sc['Δstd %'].abs().idxmax()}) "
      "-> cần cân nhắc thang đo bền.")
print("\nSo sánh 3 phương án chuẩn hóa theo 2 tiêu chí:")
print("  (1) cân bằng: tỷ số đóng góp lớn nhất / nhỏ nhất giữa các thuộc tính, tính trên 194 khách thường (lý tưởng = 1)")
print("  (2) lấn át: tỷ trọng bình phương khoảng cách thuộc về các cặp có ≥ 1 khách cực đoan (số cặp này chỉ chiếm 5.9%)")
OPTS = {
    "standard (fit 200)": (T.mean(), T.std(ddof=0)),
    "robust (median, IQR/1.349)": (T.median(), ((T.quantile(.75) - T.quantile(.25)) / 1.349).replace(0, np.nan).fillna(T.std(ddof=0))),
    "standard (fit 194 khách thường)": (T[~ext].mean(), T[~ext].std(ddof=0)),
}
ii, jj = np.triu_indices(len(T), k=1); pe = ext.values[ii] | ext.values[jj]
opt_rows = []
for name, (cen, scl) in OPTS.items():
    Zo = (T - cen) / scl
    v = Zo[~ext].var(ddof=0)
    D = pdist(Zo.values) ** 2
    opt_rows.append({"phương án": name, "đóng góp max/min (khách thường)": v.max() / v.min(),
                     "thuộc tính bị giảm trọng số nhất": v.idxmin(), "var nhỏ nhất": v.min(),
                     "tỷ trọng khoảng cách của cặp cực đoan": D[pe].sum() / D.sum()})
opt_tbl = pd.DataFrame(opt_rows).set_index("phương án")
print(opt_tbl.round(3).to_string())
SCALER = "standard (fit 194 khách thường)"
print(f"-> Chọn: {SCALER}.")
print("   Robust (IQR) làm lệch trọng số: thuộc tính có phân bố bẹt/hai đỉnh bị đánh giá phân tán thấp hơn thực tế,")
print("   city_tier (3 giá trị) có IQR thô -> đóng góp không đều. Standard fit trên toàn bộ thì std của annual_income bị")
print("   6 khách cực đoan thổi phồng 31% -> nén thu nhập của 194 khách còn lại.")
print("   Fit trên 194 khách thường: mọi thuộc tính có phương sai 1 với nhóm khách thường (đóng góp đều),")
print("   6 khách cực đoan vẫn ở xa - đúng vì họ thật sự khác biệt - và phép biến đổi áp dụng như nhau cho mọi bản ghi.")
CENTER, SCALE = OPTS[SCALER]

# ============================================================================ D3. Mã hóa
section("D3. MÃ HÓA THUỘC TÍNH PHÂN LOẠI")
print("city_tier: thứ bậc (1 < 2 < 3) -> giữ dạng số, chuẩn hóa như thuộc tính số.")
print("Định danh -> one-hot. Vấn đề: one-hot thô cho mỗi thuộc tính k mức thành k cột 0/1, đóng góp vào khoảng cách")
print("không tương xứng với 1 thuộc tính số đã chuẩn hóa. Cân bằng theo ĐÓNG GÓP KỲ VỌNG vào bình phương khoảng cách:")
print("  - thuộc tính số chuẩn hóa: E[(x_i - x_j)²] = 2·Var = 2")
print("  - one-hot nhân trọng số w: E = 2·w²·(1 - Σp_k²)  (2 cột khác nhau khi hai khách khác mức)")
print("  -> w = 1 / sqrt(1 - Σp_k²) để mỗi thuộc tính, dù số hay định danh, đóng góp kỳ vọng như nhau.")
WEIGHTS = {}
for c in KEEP_NOM:
    p = df[c].value_counts(normalize=True)
    WEIGHTS[c] = float(1 / np.sqrt(1 - (p ** 2).sum()))
    print(f"  {c}: tần suất {p.round(3).to_dict()} -> w = {WEIGHTS[c]:.3f}")


def build(d):
    Z = (transform(d) - CENTER) / SCALE
    for c in KEEP_NOM:
        oh = pd.get_dummies(d[c], prefix=c, dtype=float) * WEIGHTS[c]
        Z = pd.concat([Z, oh], axis=1)
    return Z


X = build(df)
NUM_COLS = NUM + KEEP_ORD
print(f"\nMa trận chuẩn bị: {X.shape[0]} × {X.shape[1]} ({len(NUM_COLS)} số/thứ bậc + "
      f"{X.shape[1] - len(NUM_COLS)} cột one-hot từ {len(KEEP_NOM)} thuộc tính định danh)")


def contribution(M, groups):
    """Tỷ trọng đóng góp của từng thuộc tính vào bình phương khoảng cách cặp trung bình = 2·Σ Var(cột)."""
    v = M.var(ddof=0)
    s = pd.Series({g: v[cols].sum() for g, cols in groups.items()})
    return s / s.sum()


groups = {c: [c] for c in NUM_COLS} | {c: [k for k in X.columns if k.startswith(c + "_")] for c in KEEP_NOM}
contrib_after = contribution(X, groups)
contrib_bulk = contribution(X[~ext.values], groups)
print("Tỷ trọng đóng góp vào khoảng cách - 194 khách thường:", contrib_bulk.round(3).to_dict())
print("                                    - cả 200 khách:  ", contrib_after.round(3).to_dict())

# ============================================================================ D5. PCA
section("D5. GIẢM CHIỀU BẰNG PCA (phương án thay thế để so sánh ở Câu e)")
pca_full = PCA(random_state=SEED).fit(X)
evr = pca_full.explained_variance_ratio_; cum = evr.cumsum()
# Phân tích song song (parallel analysis): giữ thành phần có trị riêng lớn hơn phân vị 95% của dữ liệu xáo trộn
rng = np.random.default_rng(SEED)
perm_eig = []
for _ in range(100):
    Xp = np.column_stack([rng.permutation(X.iloc[:, j].values) for j in range(X.shape[1])])
    perm_eig.append(PCA().fit(Xp).explained_variance_)
thr95 = np.percentile(perm_eig, 95, axis=0)
k_pa = int((pca_full.explained_variance_ > thr95).sum())
k_80 = int(np.searchsorted(cum, 0.80) + 1)
print("Phương sai giải thích (tích lũy):", {f"PC{i+1}": round(float(c), 3) for i, c in enumerate(cum)})
print(f"Số thành phần: phân tích song song -> {k_pa}; ngưỡng 80% phương sai -> {k_80}")
def pres(k):
    Pk = PCA(n_components=k, random_state=SEED).fit_transform(X)
    dk = pdist(Pk)
    nk = NearestNeighbors(n_neighbors=11).fit(Pk).kneighbors(return_distance=False)
    keep = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn_full[:, 1:], nk[:, 1:])])
    return dict(K=k, phương_sai=cum[k - 1], tương_quan_khoảng_cách=np.corrcoef(d_full, dk)[0, 1],
                giữ_10_láng_giềng=keep)


d_full = pdist(X.values)
nn_full = NearestNeighbors(n_neighbors=11).fit(X.values).kneighbors(return_distance=False)
ptab = pd.DataFrame([pres(k) for k in sorted({k_pa, 3, 5, k_80, 9})]).set_index("K")
print("Mức bảo toàn cấu trúc theo số thành phần (so với không gian đầy đủ 14 chiều):")
print(ptab.round(3).to_string())
K = k_pa
pca = PCA(n_components=K, random_state=SEED).fit(X)
P = pd.DataFrame(pca.transform(X), columns=[f"PC{i+1}" for i in range(K)])
print(f"K = {K} theo phân tích song song chỉ giữ {cum[K-1]:.1%} phương sai và {ptab.at[K, 'giữ_10_láng_giềng']:.0%} láng giềng gần;")
print(f"K = {k_80} (80% phương sai) giữ {ptab.at[k_80, 'giữ_10_láng_giềng']:.0%} láng giềng nhưng chỉ giảm 14 -> {k_80} chiều.")
print("-> KHÔNG dùng PCA làm đầu vào chính: 14 chiều với n = 200 không phải là chiều cao, và PCA làm mất cấu trúc cục bộ")
print(f"   mà phân cụm cần. Lưu PCA {K} chiều để TRỰC QUAN HÓA và làm phương án so sánh ở Câu e.")
load = pd.DataFrame(pca.components_.T, index=X.columns, columns=P.columns)
print("Hệ số tải (|loading| ≥ 0.3 được coi là đóng góp chính):")
print(load.round(2).to_string())

# ============================================================================ D6. Giảm số đối tượng
section("D6. GIẢM SỐ ĐỐI TƯỢNG")
dup = X.round(6).duplicated().sum()
print(f"n = {len(X)}; bản ghi trùng lặp hoàn toàn sau chuẩn bị: {dup}.")
print("Không lấy mẫu / không gom nhóm: 200 bản ghi nhỏ, mọi thuật toán phân cụm chạy tức thì; lấy mẫu có nguy cơ")
print(f"làm mất nhóm nhỏ như {ext.sum()} khách số đơn cực đoan (3% dữ liệu). Không loại bản ghi nào.")

# ============================================================================ D7. Phân tích thay đổi
section("D7. PHÂN TÍCH THAY ĐỔI TRƯỚC / SAU CHUẨN BỊ")
raw_num = df[NUM_ALL].astype(float)
contrib_raw = raw_num.var(ddof=0) / raw_num.var(ddof=0).sum()
zs_raw = (raw_num - raw_num.mean()) / raw_num.std(ddof=0)
contrib_z = zs_raw.var(ddof=0) / zs_raw.var(ddof=0).sum()
cmp_c = pd.DataFrame({"thô (chưa chuẩn hóa)": contrib_raw, "chỉ z-score": contrib_z}).reindex(NUM_ALL + KEEP_ORD + KEEP_NOM)
cmp_c["sau chuẩn bị"] = contrib_after.reindex(cmp_c.index)
print("Tỷ trọng đóng góp vào bình phương khoảng cách:")
print(cmp_c.fillna(0).round(4).to_string())

# Ảnh hưởng của nhóm cực đoan lên khoảng cách: ở dữ liệu chỉ z-score, các cặp có 1 điểm cực đoan chiếm bao nhiêu khoảng cách?
def share_ext(M):
    D = pdist(M) ** 2
    i, j = np.triu_indices(len(M), k=1)
    e = ext.values
    return D[e[i] | e[j]].sum() / D.sum(), (e[i] | e[j]).mean()


s_z, frac = share_ext(zs_raw[[c for c in NUM_ALL]].values)
s_x, _ = share_ext(X.values)
print(f"\nCặp có ≥ 1 khách cực đoan chiếm {frac:.1%} số cặp; chiếm {s_z:.1%} tổng bình phương khoảng cách khi chỉ z-score "
      f"-> {s_x:.1%} sau chuẩn bị (vẫn xa nhưng không còn lấn át)")


def hopkins(Z, reps=50, seed=SEED):
    Z = np.asarray(Z); rng = np.random.default_rng(seed); n, d = Z.shape; m = max(5, n // 10)
    nnm = NearestNeighbors(n_neighbors=2).fit(Z); lo, hi = Z.min(0), Z.max(0); out = []
    for _ in range(reps):
        u = nnm.kneighbors(rng.uniform(lo, hi, (m, d)), n_neighbors=1)[0].ravel()
        w = nnm.kneighbors(Z[rng.choice(n, m, replace=False)], n_neighbors=2)[0][:, 1]
        out.append(u.sum() / (u.sum() + w.sum()))
    return np.mean(out), np.std(out)


hz = hopkins(zs_raw.values); hx = hopkins(X.values); hp = hopkins(P.values)
print(f"Hopkins (0.5 = ngẫu nhiên, → 1 = có cụm): chỉ z-score {hz[0]:.3f} ± {hz[1]:.3f}; "
      f"sau chuẩn bị {hx[0]:.3f} ± {hx[1]:.3f}; PCA {K} chiều {hp[0]:.3f} ± {hp[1]:.3f}")
print("Lưu ý: Hopkins cao ở dữ liệu chỉ z-score một phần do vài điểm cực đoan làm khung lấy mẫu đều rất rộng, "
      "không nhất thiết là cấu trúc cụm tốt hơn.")
corr_after = X[NUM_COLS].corr(method="pearson")
off = corr_after.where(~np.eye(len(corr_after), dtype=bool)).abs()
print(f"|Pearson| lớn nhất giữa các thuộc tính số sau chuẩn bị: {off.max().max():.2f} "
      f"({off.stack().idxmax()}) - không còn cặp nào > 0.9")
skew_tbl = pd.DataFrame({"skew trước": df[NUM].skew(), "skew sau": X[NUM].skew()}).round(2)
print("Độ lệch trước/sau:"); print(skew_tbl.T.to_string())

# ============================================================================ Lưu
section("LƯU KẾT QUẢ")
out = pd.concat([df[["customer_id"]], X.round(6)], axis=1)
out.to_csv(os.path.join(DATA, "05_prepared.csv"), index=False)
pd.concat([df[["customer_id"]], P.round(6)], axis=1).to_csv(os.path.join(DATA, "05_prepared_pca.csv"), index=False)
# Biến thể độ nhạy (Câu c): AOV trước bước A3 đối soát nhiễu, cùng tham số biến đổi/chuẩn hóa
ch = pd.read_csv(os.path.join(DATA, "04_change_log.csv"))
a3 = ch[ch.step == "A3"].set_index("customer_id")["old"]
d03 = df.copy()
m = d03["customer_id"].isin(a3.index)
d03.loc[m, "avg_order_value"] = d03.loc[m, "customer_id"].map(a3).values
pd.concat([df[["customer_id"]], build(d03).round(6)], axis=1).to_csv(os.path.join(DATA, "05_prepared_aov03.csv"), index=False)
params = dict(dropped=["customer_id", "total_spend_12m", "annual_income_was_missing"] +
              [c for c in NOMINAL + ORDINAL if not summ.at[c, "giữ"]],
              transforms=TRANS, scaler=SCALER, center=CENTER.round(6).to_dict(), scale=SCALE.round(6).to_dict(),
              onehot_weights=WEIGHTS, pca_components=K, pca_explained=round(float(cum[K - 1]), 4))
json.dump(params, open(os.path.join(HERE, "prep_params.json"), "w"), ensure_ascii=False, indent=2)
print(f"data/05_prepared.csv: {out.shape}; data/05_prepared_pca.csv: {P.shape[0]} × {K}; "
      f"data/05_prepared_aov03.csv: {m.sum()} AOV khác 05_prepared; cau_d/prep_params.json")

# ============================================================================ Hình
fig, ax = plt.subplots(figsize=(12, 4.6))
idx = cmp_c.index.tolist(); xx = np.arange(len(idx)); w = 0.27
for k, (col, cl) in enumerate(zip(cmp_c.columns, [GREY, ORANGE, BLUE])):
    ax.bar(xx + (k - 1) * w, cmp_c[col].fillna(0), w, color=cl, label=col)
ax.axhline(1 / len(groups), color=BLUE, ls="--", lw=.8, label=f"chia đều (1/{len(groups)})")
ax.set_xticks(xx, idx, rotation=30, ha="right")
ax.set_ylabel("Tỷ trọng đóng góp vào khoảng cách")
ax.set_title("Đóng góp của từng thuộc tính vào khoảng cách Euclid: dữ liệu thô bị total_spend chiếm gần 100%")
ax.legend(fontsize=8.5)
fig.tight_layout(); save(fig, "d_feature_contribution.png")

fig, axes = plt.subplots(2, 5, figsize=(19, 6.6))
for ax, c in zip(axes.ravel(), NUM + KEEP_ORD):
    ax.hist(X[c], bins=25, color=BLUE, edgecolor="white")
    ax.set_title(f"{c}\n{TRANS.get(c, 'thứ bậc')} + chuẩn hóa; skew {df[c].skew():.2f} → {X[c].skew():.2f}", fontsize=9.5)
fig.suptitle("Phân bố thuộc tính số sau chuẩn bị (data/05_prepared.csv)", fontsize=12)
fig.tight_layout(); save(fig, "d_distributions_after.png")

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1, 1.1, 1.3]})
ax = axes[0]
ax.bar(range(1, len(evr) + 1), pca_full.explained_variance_, color=BLUE, label="trị riêng")
ax.plot(range(1, len(evr) + 1), thr95, color=RED, marker="o", ms=3, label="phân vị 95% dữ liệu xáo trộn")
ax.axvline(K + .5, color="black", ls="--", lw=.8)
ax.set_xlabel("Thành phần"); ax.set_ylabel("Trị riêng")
ax.set_title(f"Phân tích song song: giữ K = {K} ({cum[K-1]:.0%} phương sai)"); ax.legend(fontsize=8)
ax = axes[1]
im = ax.imshow(load.values, cmap="RdBu_r", vmin=-.7, vmax=.7, aspect="auto")
ax.set_yticks(range(len(load)), load.index, fontsize=8); ax.set_xticks(range(K), load.columns)
for i in range(load.shape[0]):
    for j in range(K):
        if abs(load.iat[i, j]) >= .3:
            ax.text(j, i, f"{load.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=.8); ax.set_title("Hệ số tải (ghi số khi |tải| ≥ 0.3)")
ax = axes[2]
ax.scatter(P["PC1"], P["PC2"], s=18, color=BLUE, alpha=.6, label="khách")
ax.scatter(P.loc[ext.values, "PC1"], P.loc[ext.values, "PC2"], s=170, marker="*", color="black",
           label=f"orders > {fence:g} (n={ext.sum()})")
ax.set_xlabel(f"PC1 ({evr[0]:.0%})"); ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
ax.set_title(f"PC1-PC2; Hopkins sau chuẩn bị = {hx[0]:.2f}"); ax.legend(fontsize=8)
fig.tight_layout(); save(fig, "d_pca.png")

fig, ax = plt.subplots(figsize=(8.5, 7))
cm = corr_after.values
im = ax.imshow(np.ma.masked_array(cm, np.triu(np.ones_like(cm, dtype=bool), 1)), cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(cm)):
    for j in range(i + 1):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(cm[i, j]) > .6 else "black")
ax.set_xticks(range(len(cm)), corr_after.columns, rotation=45, ha="right"); ax.set_yticks(range(len(cm)), corr_after.index)
ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, shrink=.75, label="Pearson")
ax.set_title("Tương quan Pearson sau chuẩn bị (đã bỏ total, đã log)")
fig.tight_layout(); save(fig, "d_correlation_after.png")
print("\nXong.")
