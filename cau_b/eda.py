"""
Câu b - Data Understanding.
Chỉ đọc data/03_raw_final.csv (không dùng injection_log, file 01 hay true_segment).
Lưu hình vào figures/ với tiền tố b_, in các bảng số liệu ra màn hình.

  1. Tổng quan kiểu dữ liệu
  2. Vi phạm miền giá trị (quy tắc nghiệp vụ)
  3. Phân bố và độ lệch của thuộc tính số
  4. Ma trận tương quan
  5. Tỷ lệ thiếu và kiểm tra việc thiếu có phụ thuộc thuộc tính khác
  6. Ngoại lai theo IQR (thang thô và thang log10)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
DPI = 200
BLUE, ORANGE, RED, GREY = "#2a6f97", "#e07a5f", "#d62828", "#adb5bd"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)

df = pd.read_csv(os.path.join(ROOT, "data", "03_raw_final.csv"))

NUM = ["age", "tenure_months", "annual_income", "orders_12m", "avg_order_value", "total_spend_12m",
       "recency_days", "discount_usage_rate", "return_rate", "avg_session_minutes"]
ORD = ["city_tier"]
CAT = ["gender", "preferred_category", "payment_method"]
LOGX = ["annual_income", "orders_12m", "avg_order_value", "total_spend_12m"]   # đại lượng dương, lệch phải


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[hình] figures/{name}")


# ============================================================================ 1. Tổng quan
section("1. TỔNG QUAN")
print(f"Kích thước: {df.shape[0]} bản ghi × {df.shape[1]} cột (1 ID + {df.shape[1]-1} thuộc tính đầu vào)")
print(f"customer_id trùng: {df['customer_id'].duplicated().sum()}")
kind = {c: "số" for c in NUM} | {c: "thứ bậc" for c in ORD} | {c: "định danh" for c in CAT}
print(pd.DataFrame({"kiểu": [kind[c] for c in NUM + ORD + CAT],
                    "dtype": df[NUM + ORD + CAT].dtypes.astype(str).values,
                    "n_unique": df[NUM + ORD + CAT].nunique().values},
                   index=NUM + ORD + CAT).to_string())
for c in CAT + ORD:
    print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}")

# ============================================================================ 2. Vi phạm miền giá trị
section("2. VI PHẠM MIỀN GIÁ TRỊ (quy tắc nghiệp vụ, không phụ thuộc ngưỡng thống kê)")
ratio = df["total_spend_12m"] / (df["orders_12m"] * df["avg_order_value"])
RULES = [  # (tên, cột bị coi là sai, mặt nạ vi phạm, lý do)
    ("age ngoài [18, 100]", "age", ~df["age"].between(18, 100), "tài khoản TMĐT yêu cầu ≥ 18 tuổi"),
    ("tenure_months < 0", "tenure_months", df["tenure_months"] < 0, "thời gian thành viên không âm"),
    ("avg_order_value ≤ 0", "avg_order_value", df["avg_order_value"] <= 0, "giá trị đơn phải dương"),
    ("recency_days ngoài [0, 365]", "recency_days", ~df["recency_days"].between(0, 365),
     "orders_12m ≥ 1 nên lần mua gần nhất phải trong 12 tháng"),
    ("discount_usage_rate ngoài [0, 1]", "discount_usage_rate", ~df["discount_usage_rate"].between(0, 1), "là tỷ lệ"),
    ("return_rate ngoài [0, 1]", "return_rate", ~df["return_rate"].between(0, 1), "là tỷ lệ"),
    ("avg_session_minutes > 10080", "avg_session_minutes", df["avg_session_minutes"] > 10080, "1 tuần = 10 080 phút"),
    ("total/(orders×AOV) ngoài [0.5, 2] (AOV > 0)", "total_spend_12m",
     ~ratio.between(0.5, 2) & (df["avg_order_value"] > 0), "tổng chi tiêu ≈ số đơn × giá trị đơn"),
]
viol_rows = []
for name, col, m, why in RULES:
    vals = ratio[m] if name.startswith("total") else df.loc[m, col]
    viol_rows.append(dict(quy_tắc=name, n=int(m.sum()), customer_id=", ".join(df.loc[m, "customer_id"]),
                          giá_trị=", ".join(f"{x:g}" for x in vals), lý_do=why))
viol = pd.DataFrame(viol_rows)
print(viol.to_string(index=False))
any_viol = np.column_stack([m for _, _, m, _ in RULES]).any(axis=1)
print(f"\nTổng số bản ghi vi phạm ≥ 1 quy tắc: {any_viol.sum()}")

# Bản dùng để MÔ TẢ thống kê: đặt NaN đúng các ô vi phạm (không xóa dòng), để lỗi không làm méo thống kê.
dv = df.copy()
for _, col, m, _ in RULES:
    dv.loc[m, col] = np.nan
print(f"Bản mô tả dv: {int(dv[NUM].isna().sum().sum() - df[NUM].isna().sum().sum())} ô vi phạm được đặt NaN "
      f"(chỉ dùng cho các mục 3-6, KHÔNG phải tiền xử lý)")

# ============================================================================ 3. Phân bố và độ lệch
section("3. PHÂN BỐ VÀ ĐỘ LỆCH CỦA THUỘC TÍNH SỐ (trên dv)")
rows = []
for c in NUM:
    x = dv[c].dropna()
    r = dict(n=len(x), mean=x.mean(), std=x.std(), min=x.min(), Q1=x.quantile(.25), median=x.median(),
             Q3=x.quantile(.75), max=x.max(), skew=x.skew(), max_trên_median=x.max() / x.median() if x.median() else np.nan)
    r["skew_log10"] = stats.skew(np.log10(x)) if c in LOGX else np.nan
    rows.append(pd.Series(r, name=c))
dist = pd.DataFrame(rows)
with pd.option_context("display.float_format", lambda v: f"{v:,.3f}" if abs(v) < 1e4 else f"{v:,.3g}"):
    print(dist.to_string())
print("\nĐộ chênh thang đo giữa các thuộc tính (độ lệch chuẩn):")
print("  ", dv[NUM].std().sort_values(ascending=False).map(lambda v: f"{v:.3g}").to_dict())
print("Giá trị dồn ở biên: discount_usage_rate = 0:", int((df["discount_usage_rate"] == 0).sum()),
      "| = 1:", int((df["discount_usage_rate"] == 1).sum()))

fig, axes = plt.subplots(2, 5, figsize=(21, 7.8))
for ax, c in zip(axes.ravel(), NUM):
    x = dv[c].dropna()
    if c in LOGX:
        ax.hist(x, bins=np.logspace(np.log10(x.min()), np.log10(x.max()), 25), color=BLUE, edgecolor="white")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(f"{c} (thang log)")
        ax.set_title(f"{c}\nskew = {x.skew():.2f} (log10: {stats.skew(np.log10(x)):.2f})", fontsize=10)
    else:
        ax.hist(x, bins=25, color=BLUE, edgecolor="white")
        ax.set_xlabel(c)
        ax.set_title(f"{c}\nskew = {x.skew():.2f}", fontsize=10)
    ax.set_ylabel("Số bản ghi")
fig.suptitle("Phân bố thuộc tính số - file 03, đã bỏ các ô vi phạm miền giá trị", fontsize=13)
fig.tight_layout()
save(fig, "b_distributions.png")

fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
for ax, c in zip(axes, ORD + CAT):
    vc = df[c].astype("string").fillna("NaN").value_counts()
    ax.bar(vc.index, vc.values, color=[GREY if i == "NaN" else BLUE for i in vc.index])
    ax.set_title(c)
    ax.tick_params(axis="x", rotation=25)
fig.suptitle("Phân bố thuộc tính phân loại (xám = thiếu)", fontsize=13)
fig.tight_layout()
save(fig, "b_categorical.png")

# ============================================================================ 4. Tương quan
section("4. MA TRẬN TƯƠNG QUAN (Spearman - dựa trên hạng, bền với độ lệch và ngoại lai)")
corr = dv[NUM + ORD].corr(method="spearman")
pear = dv[NUM].corr(method="pearson")
print(corr.round(2).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(10).to_frame("Spearman")
top["Pearson (thô)"] = [pear.loc[a, b] if a in pear.index and b in pear.columns else np.nan for a, b in top.index]
print("\n10 cặp tương quan mạnh nhất:")
print(top.round(3).to_string())

fig, ax = plt.subplots(figsize=(9.5, 8))
mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
im = ax.imshow(np.ma.masked_array(corr.values, mask), cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(corr)):
    for j in range(i + 1):
        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.index)
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
ax.set_title("Ma trận tương quan Spearman (file 03, đã bỏ ô vi phạm miền)")
fig.tight_layout()
save(fig, "b_correlation.png")

# ---------------------------------------------------------------------------- 4b. Dư thừa do quan hệ nhân
section("4b. DƯ THỪA total ≈ orders × AOV: tương quan từng cặp KHÔNG đo đúng, cần hồi quy trên thang log")
spend = dv[["orders_12m", "avg_order_value", "total_spend_12m"]].dropna()
L = np.log10(spend)
print(f"n = {len(L)} bản ghi (đã bỏ các ô vi phạm miền, gồm AOV âm và total sai đơn vị)")


def ols(y, X):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    return beta, 1 - res.var() / y.var(), res.std(ddof=A.shape[1])


beta, r2, se = ols(L["total_spend_12m"].values, L[["orders_12m", "avg_order_value"]].values)
print(f"log10(total) = {beta[0]:.3f} + {beta[1]:.3f}·log10(orders) + {beta[2]:.3f}·log10(AOV)")
print(f"  R² = {r2:.4f}, độ lệch chuẩn phần dư = {se:.4f} (log10) ≈ ±{(10**se - 1):.1%} trên thang gốc")
print("  -> hệ số ≈ 1 đúng như quan hệ total = orders × AOV; total gần như là tổ hợp tuyến tính (trên thang log) của 2 thuộc tính kia")
r2_o = ols(L["total_spend_12m"].values, L[["orders_12m"]].values)[1]
r2_a = ols(L["total_spend_12m"].values, L[["avg_order_value"]].values)[1]
print(f"  So sánh: R² chỉ với log(orders) = {r2_o:.3f}; chỉ với log(AOV) = {r2_a:.3f}; "
      f"ρ Spearman (orders, total) = {spend.corr('spearman').iat[0, 2]:.3f}, (AOV, total) = {spend.corr('spearman').iat[1, 2]:.3f}")
vif = {c: 1 / (1 - ols(L[c].values, L.drop(columns=c).values)[1]) for c in L.columns}
print("  VIF trên thang log:", {k: round(float(v), 1) for k, v in vif.items()}, "(VIF > 10 = đa cộng tuyến nghiêm trọng)")
print(f"  ρ Spearman (orders, AOV) = {spend.corr('spearman').iat[0, 1]:.3f} -> hai thành phần gần độc lập, mỗi cái mang thông tin riêng")

# Nếu CHỈ giữ total: mất chiều "phong cách mua" = log(orders) - log(AOV) (vuông góc với log(total))
style = (L["orders_12m"] - L["avg_order_value"]).rename("style")
print(f"\nChiều bị mất nếu chỉ giữ total: style = log10(orders) - log10(AOV)")
print(f"  std(style) = {style.std():.3f} vs std(log10 total) = {L['total_spend_12m'].std():.3f}; "
      f"ρ(style, log total) = {stats.spearmanr(style, L['total_spend_12m'])[0]:.3f}")
beh = dv.loc[style.index, ["discount_usage_rate", "avg_session_minutes", "recency_days", "return_rate"]]
cmp = pd.DataFrame({
    "ρ với log10(total)": [stats.spearmanr(L["total_spend_12m"], beh[c], nan_policy="omit")[0] for c in beh],
    "ρ với style": [stats.spearmanr(style, beh[c], nan_policy="omit")[0] for c in beh]}, index=beh.columns)
print("  Tương quan Spearman của các hành vi khác với total và với style:")
print(cmp.round(3).to_string())
print("  -> discount_usage_rate gần như không liên quan đến total nhưng liên quan rõ đến style: thông tin này CHỈ nằm trong style")
lt, st = L["total_spend_12m"].values, style.values
i, j = np.triu_indices(len(lt), k=1)
same_total = np.abs(lt[i] - lt[j]) < np.log10(1.10)          # tổng chi tiêu chênh < 10%
diff_style = np.abs(st[i] - st[j]) > 1                         # tỷ số orders/AOV chênh > 10 lần
print(f"  Cặp khách có tổng chi tiêu chênh < 10%: {same_total.sum()}; trong đó {(same_total & diff_style).sum()} cặp "
      f"({(same_total & diff_style).sum() / same_total.sum():.1%}) có tỷ số orders/AOV chênh > 10 lần "
      f"-> chỉ dùng total sẽ coi các cặp này là gần nhau")
k = int(np.argmax(np.where(same_total & diff_style, np.abs(st[i] - st[j]), -1)))
a_, b_ = style.index[i[k]], style.index[j[k]]
print("  Ví dụ:", " | ".join(f"{df.at[x, 'customer_id']}: {int(df.at[x, 'orders_12m'])} đơn × "
                             f"{df.at[x, 'avg_order_value']/1e3:.0f}k = {df.at[x, 'total_spend_12m']/1e6:.2f} triệu" for x in (a_, b_)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
ax = axes[0]
pred = beta[0] + beta[1] * L["orders_12m"] + beta[2] * L["avg_order_value"]
ax.scatter(pred, L["total_spend_12m"], s=16, alpha=0.6, color=BLUE)
mn, mx = pred.min(), pred.max()
ax.plot([mn, mx], [mn, mx], color=RED, lw=1)
ax.set_xlabel("log10(total) dự đoán từ log10(orders) + log10(AOV)")
ax.set_ylabel("log10(total_spend_12m) thực tế")
ax.set_title(f"Hồi quy trên thang log: R² = {r2:.3f}, hệ số {beta[1]:.2f} và {beta[2]:.2f}\n"
             f"(trong khi ρ(AOV, total) chỉ {spend.corr('spearman').iat[1, 2]:.2f})", fontsize=10)
ax = axes[1]
sc = ax.scatter(L["orders_12m"], L["avg_order_value"], c=L["total_spend_12m"], cmap="viridis", s=18)
ylo, yhi = L["avg_order_value"].min() - 0.1, L["avg_order_value"].max() + 0.15
xlo, xhi = L["orders_12m"].min() - 0.1, L["orders_12m"].max() + 0.1
for lvl in range(6, 10):
    xs = np.linspace(xlo, xhi, 50)
    ax.plot(xs, lvl - xs, color="0.5", lw=0.7, ls="--")
    xt = lvl - (ylo + 0.08)                       # điểm đường cắt gần đáy khung
    if xlo < xt < xhi:
        ax.text(xt, ylo + 0.08, f"10^{lvl} VND", fontsize=7.5, color="0.35", rotation=-35, ha="right", va="bottom")
for x, mk in ((a_, "o"), (b_, "s")):
    ax.scatter(L.at[x, "orders_12m"], L.at[x, "avg_order_value"], s=120, facecolors="none", edgecolors=RED, lw=1.6)
    ax.annotate(df.at[x, "customer_id"], (L.at[x, "orders_12m"], L.at[x, "avg_order_value"]),
                xytext=(6, 6), textcoords="offset points", fontsize=8, color=RED)
ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
fig.colorbar(sc, ax=ax, label="log10(total)")
ax.set_xlabel("log10(orders_12m)"); ax.set_ylabel("log10(avg_order_value)")
ax.set_title(f"Đường đứt = cùng tổng chi tiêu. Chỉ giữ total thì mất vị trí dọc đường\n"
             f"(nhiều đơn rẻ ↔ ít đơn đắt), vd {df.at[a_, 'customer_id']} và {df.at[b_, 'customer_id']} (đỏ) thành 'giống nhau'", fontsize=10)
fig.tight_layout()
save(fig, "b_spend_redundancy.png")

# ============================================================================ 5. Giá trị thiếu
section("5. GIÁ TRỊ THIẾU")
miss = df.isna().sum()
MCOLS = miss[miss > 0].index.tolist()
print(pd.DataFrame({"n_thiếu": miss[MCOLS], "%": (miss[MCOLS] / len(df) * 100).round(1)}).to_string())
n_any = int(df.isna().any(axis=1).sum())
print(f"Bản ghi có ≥ 1 ô thiếu: {n_any} ({n_any/len(df):.1%}) -> xóa theo dòng sẽ mất ngần ấy dữ liệu")
print(f"Cộng thêm {any_viol.sum()} bản ghi có lỗi miền: tổng bản ghi 'không sạch' = "
      f"{int((df.isna().any(axis=1) | any_viol).sum())}")

print("\n5.1 Sàng lọc: thuộc tính khác có phân bố khác nhau giữa nhóm THIẾU và KHÔNG THIẾU?")
print("    Mann-Whitney U cho thuộc tính số, χ² cho thuộc tính phân loại; Bonferroni trên toàn bộ phép thử.")
tests = []
for mc in MCOLS:
    ind = df[mc].isna()
    for c in NUM + ORD + CAT:
        if c == mc:
            continue
        if c in NUM:
            a, b = dv.loc[ind, c].dropna(), dv.loc[~ind, c].dropna()
            p = stats.mannwhitneyu(a, b).pvalue
            d = f"trung vị thiếu {a.median():.3g} vs không thiếu {b.median():.3g}"
        else:
            ct = pd.crosstab(ind, df[c].astype("string").fillna("NaN"))
            p, d = stats.chi2_contingency(ct)[1], "χ²"
        tests.append(dict(thiếu=mc, so_với=c, p=p, mô_tả=d))
tests = pd.DataFrame(tests)
tests["p_Bonferroni"] = (tests["p"] * len(tests)).clip(upper=1)
print(f"    Tổng số phép thử: {len(tests)}. 6 kết quả p nhỏ nhất:")
print(tests.sort_values("p").head(6).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n5.2 Kiểm định có định hướng: tỷ lệ thiếu có xu hướng theo nhóm tuổi? (Cochran-Armitage)")
AGE_BINS, AGE_LAB = [17, 27, 39, 100], ["18-27", "28-39", "40+"]
age_grp = pd.cut(dv["age"], AGE_BINS, labels=AGE_LAB)
n_grp = age_grp.value_counts().reindex(AGE_LAB)


def cochran_armitage(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    t = np.arange(len(k)); N = n.sum(); pb = k.sum() / N
    T = (t * (k - n * pb)).sum()
    var = pb * (1 - pb) * ((n * t ** 2).sum() - (n * t).sum() ** 2 / N)
    z = T / np.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


trend = []
for mc in MCOLS:
    k = df[mc].isna().groupby(age_grp, observed=True).sum().reindex(AGE_LAB).values
    z, p = cochran_armitage(k, n_grp.values)
    trend.append(dict(thuộc_tính=mc, **{f"{l} (k/n)": f"{kk}/{nn} = {kk/nn:.1%}" for l, kk, nn in zip(AGE_LAB, k, n_grp)},
                      z=round(z, 2), p=p))
trend = pd.DataFrame(trend)
print(f"    (bỏ {int(age_grp.isna().sum())} bản ghi có tuổi vi phạm miền)")
print(trend.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

fig, axes = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
order = np.argsort(dv["age"].fillna(dv["age"].median()).values, kind="stable")
ax.imshow(df[MCOLS].isna().T.values[:, order].astype(int), aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks(range(len(MCOLS)), MCOLS)
ax.set_xlabel("200 bản ghi, sắp theo tuổi tăng dần →")
ax.set_title("Ma trận giá trị thiếu (đen = thiếu)")
ax = axes[1]
w = 0.26
for i, mc in enumerate(MCOLS):
    k = df[mc].isna().groupby(age_grp, observed=True).sum().reindex(AGE_LAB).values
    r = k / n_grp.values
    lo, hi = np.array([wilson(kk, nn) for kk, nn in zip(k, n_grp.values)]).T
    ax.bar(np.arange(3) + (i - 1) * w, r, w, yerr=[r - lo, hi - r], capsize=3,
           color={"annual_income": ORANGE, "city_tier": BLUE}.get(mc, GREY), label=f"{mc} (p xu hướng = {trend.p[i]:.2g})")
ax.set_xticks(range(3), [f"{l}\n(n={n})" for l, n in zip(AGE_LAB, n_grp.values)])
ax.set_ylabel("Tỷ lệ thiếu (KTC Wilson 95%)")
ax.set_title("Tỷ lệ thiếu theo nhóm tuổi")
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
save(fig, "b_missing.png")

# ============================================================================ 6. Ngoại lai theo IQR
section("6. NGOẠI LAI THEO IQR (Q1 - 1.5·IQR, Q3 + 1.5·IQR) trên dv")


def iqr_bounds(s):
    q1, q3 = s.quantile([.25, .75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


out = []
flags_raw = pd.DataFrame(False, index=df.index, columns=NUM)
for c in NUM:
    s = dv[c].dropna()
    lo, hi = iqr_bounds(s)
    f = (s < lo) | (s > hi)
    flags_raw.loc[f[f].index, c] = True
    r = dict(n_thô=int(f.sum()), ngưỡng_thô=f"[{lo:.3g}, {hi:.3g}]",
             giá_trị_lớn_nhất=", ".join(f"{v:g}" for v in sorted(s[f])[-5:]))
    if c in LOGX:
        ls = np.log10(s); llo, lhi = iqr_bounds(ls); fl = (ls < llo) | (ls > lhi)
        r.update(n_log10=int(fl.sum()), ngưỡng_log10=f"[{10**llo:.3g}, {10**lhi:.3g}]")
    out.append(pd.Series(r, name=c))
print(pd.DataFrame(out).fillna("").to_string())
print(f"\nBản ghi bị IQR (thang thô) gắn cờ ở ≥ 1 thuộc tính: {int(flags_raw.any(axis=1).sum())}")

# Nhóm số đơn cực đoan: xem có nhất quán nội tại không (để phân biệt 'lỗi' và 'khách thật nhưng cực đoan')
lo, hi = iqr_bounds(dv["orders_12m"].dropna())
ext = dv["orders_12m"] > hi
srt = np.sort(dv["orders_12m"].dropna().unique()); g = np.diff(srt).argmax()
print(f"\nNhóm orders_12m > {hi:g}: {ext.sum()} bản ghi {df.loc[ext, 'customer_id'].tolist()}")
print(f"  Khoảng trống lớn nhất trong orders_12m: {srt[g]:g} → {srt[g+1]:g}")
print(f"  orders = {df.loc[ext, 'orders_12m'].tolist()}")
print(f"  AOV (triệu) = {(df.loc[ext, 'avg_order_value'] / 1e6).round(2).tolist()} "
      f"| AOV lớn nhất của phần còn lại = {dv.loc[~ext, 'avg_order_value'].max() / 1e6:.2f}")
print(f"  total/(orders×AOV) = {ratio[ext].round(3).tolist()}  -> nhất quán nội tại")
print(f"  annual_income (triệu/năm) = {df.loc[ext, 'annual_income'].tolist()}")
print(f"  recency_days = {df.loc[ext, 'recency_days'].tolist()}")
print("  IQR trên log10(orders_12m):",
      "phát hiện", int(((np.log10(dv['orders_12m']) > iqr_bounds(np.log10(dv['orders_12m']))[1])).sum()), "bản ghi")

fig, axes = plt.subplots(2, 5, figsize=(21, 7.4))
for ax, c in zip(axes.ravel(), NUM):
    s_all = df[c].dropna()
    s = dv[c].dropna()
    lo_, hi_ = iqr_bounds(s)
    bad = s_all[~s_all.index.isin(s.index)]
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(s_all))
    ax.boxplot(s, positions=[0], widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#d6e6f0", edgecolor=BLUE), medianprops=dict(color=BLUE))
    ok = s[(s >= lo_) & (s <= hi_)]; fl = s[(s < lo_) | (s > hi_)]
    ax.scatter(jit[s_all.index.get_indexer(ok.index)], ok, s=8, color=BLUE, alpha=0.35)
    ax.scatter(jit[s_all.index.get_indexer(fl.index)], fl, s=22, color=ORANGE, label="ngoại lai IQR")
    if len(bad):
        ax.scatter(jit[s_all.index.get_indexer(bad.index)], bad, s=50, marker="X", color=RED, label="vi phạm miền")
    if c in LOGX and (s_all > 0).all():
        ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_title(f"{c}\nIQR: {len(fl)}, vi phạm miền: {len(bad)}", fontsize=10)
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("Ngoại lai: boxplot (IQR thang thô) + điểm vi phạm miền giá trị - file 03 "
             "(trục log cho thuộc tính lệch phải nếu không có giá trị ≤ 0)", fontsize=12)
fig.tight_layout()
save(fig, "b_outliers_iqr.png")

fig, ax = plt.subplots(figsize=(8, 5.6))
v = dv.dropna(subset=["orders_12m", "avg_order_value"])
e = v["orders_12m"] > hi
ax.scatter(v.loc[~e, "orders_12m"], v.loc[~e, "avg_order_value"] / 1e6, s=20, alpha=0.6, color=BLUE,
           label=f"orders ≤ {hi:g} (n={(~e).sum()})")
ax.scatter(v.loc[e, "orders_12m"], v.loc[e, "avg_order_value"] / 1e6, s=180, marker="*", color="black",
           label=f"orders > {hi:g} - ngoại lai IQR (n={e.sum()})")
ax.set_xscale("log"); ax.set_yscale("log")
fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}")
for a in (ax.xaxis, ax.yaxis):
    a.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    a.set_major_formatter(fmt); a.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("orders_12m (thang log)")
ax.set_ylabel("avg_order_value (triệu VND, thang log)")
ax.set_title("Nhóm số đơn cực đoan trong không gian orders × AOV (file 03)")
ax.grid(True, color="0.92")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
save(fig, "b_orders_vs_aov.png")

# ============================================================================ 7. Định lượng ảnh hưởng
section("7. ĐỊNH LƯỢNG ẢNH HƯỞNG CỦA CÁC VẤN ĐỀ (bằng chứng cho phần nhận định)")
s = df["avg_session_minutes"].dropna()
big = s > s.quantile(.75) + 1.5 * (s.quantile(.75) - s.quantile(.25))
print(f"avg_session_minutes: std = {s.std():.1f} (đủ) vs {s[~big].std():.1f} (bỏ {big.sum()} ngoại lai IQR); "
      f"mean = {s.mean():.1f} vs {s[~big].mean():.1f}")
pv = df[["orders_12m", "avg_session_minutes"]].dropna()
no_big = pv[pv["avg_session_minutes"] <= s[~big].max()]
no_both = no_big[no_big["orders_12m"] <= hi]
print("Pearson(orders_12m, avg_session_minutes): "
      f"đủ = {pv.corr().iat[0, 1]:.3f} | bỏ ngoại lai session = {no_big.corr().iat[0, 1]:.3f} | "
      f"bỏ thêm nhóm orders > {hi:g} = {no_both.corr().iat[0, 1]:.3f} | Spearman (đủ) = {pv.corr('spearman').iat[0, 1]:.3f}")
cc = df.dropna()
young = lambda x: x["age"].between(18, 27).mean()
print(f"Xóa theo dòng (listwise): còn {len(cc)}/{len(df)} bản ghi; tỷ lệ khách 18-27 tuổi: "
      f"{young(df):.1%} -> {young(cc):.1%}")
sd = dv[NUM].std()
print(f"Tỷ số độ lệch chuẩn lớn nhất / nhỏ nhất giữa các thuộc tính số: {sd.max() / sd.min():.2e} "
      f"({sd.idxmax()} / {sd.idxmin()})")
print(f"Số giá trị dồn ở biên: discount = 0: {(df['discount_usage_rate'] == 0).sum()}, = 1: "
      f"{(df['discount_usage_rate'] == 1).sum()}; recency_days = 365: {(df['recency_days'] == 365).sum()}, "
      f"= 0: {(df['recency_days'] == 0).sum()}")

print("\nXong. Hình đã lưu trong figures/ với tiền tố b_.")
