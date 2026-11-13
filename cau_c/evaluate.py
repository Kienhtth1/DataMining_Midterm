"""
Câu c - ĐÁNH GIÁ CUỐI (chạy sau preprocess.py).
Đây là file DUY NHẤT của Câu c được đọc dữ liệu sạch (01) và injection_log - chỉ để đo sai số,
không quyết định gì trong tiền xử lý.

  E1. Sai số điền giá trị thiếu: trung vị vs KNN vs Iterative (và cách được chọn)
  E2. Sai số ở các ô lỗi / nghi ngờ đã sửa
  E3. Hiệu quả đối soát nhiễu AOV (A3): phát hiện đúng bao nhiêu, sai số trước/sau
  E4. Khách mua sỉ có được giữ nguyên không
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA, FIG, HERE = os.path.join(ROOT, "data"), os.path.join(ROOT, "figures"), os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREY, GREEN = "#2a6f97", "#e07a5f", "#adb5bd", "#2a9d8f"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.width", 200)

f01 = pd.read_csv(os.path.join(DATA, "01_clean_ground_truth.csv")).set_index("customer_id")
f03 = pd.read_csv(os.path.join(DATA, "03_raw_final.csv")).set_index("customer_id")
f04 = pd.read_csv(os.path.join(DATA, "04_preprocessed.csv")).set_index("customer_id")
log = pd.read_csv(os.path.join(DATA, "injection_log.csv"))
var = pd.read_csv(os.path.join(HERE, "imputed_variants.csv"))
ch = pd.read_csv(os.path.join(DATA, "04_change_log.csv"))
METHODS = ["median", "knn", "iterative"]
legit = set(log.loc[log.type == "outlier_legit", "customer_id"])


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def iqr01(c):
    q = f01[c].quantile([.25, .75]); return q.iloc[1] - q.iloc[0]


# ============================================================================ E1
section("E1. SAI SỐ ĐIỀN GIÁ TRỊ THIẾU (ô thiếu trong 03) so với file 01")
mv = var[var.cell_type == "missing"].copy()
mv["truth"] = [f01.at[r.customer_id, r.column] for r in mv.itertuples()]
rows = []
for c, g in mv.groupby("column"):
    r = dict(column=c, n=len(g), chosen=g.chosen_method.iloc[0])
    for m in METHODS + ["chosen"]:
        e = (g[m] - g["truth"]).abs()
        r[f"MAE_{m}"] = e.mean()
        r[f"NMAE_{m}"] = e.mean() / iqr01(c)
        if c == "city_tier":
            r[f"acc_{m}"] = (g[m].round() == g["truth"]).mean()
    rows.append(r)
e1 = pd.DataFrame(rows).set_index("column")
print("MAE (thang gốc):")
print(e1[[f"MAE_{m}" for m in METHODS + ["chosen"]] + ["n", "chosen"]].round(3).to_string())
print("\nNMAE = MAE / IQR của file 01:")
print(e1[[f"NMAE_{m}" for m in METHODS + ["chosen"]]].round(3).to_string())
if "acc_median" in e1:
    print("\ncity_tier - độ chính xác:", {m: round(float(e1.at["city_tier", f"acc_{m}"]), 3) for m in METHODS + ["chosen"]},
          f"| tỷ lệ lớp phổ biến nhất trong 01: {f01['city_tier'].value_counts(normalize=True).max():.3f}")

# income theo nhóm tuổi: trung vị bị lệch hệ thống ở khách trẻ?
inc = mv[mv.column == "annual_income"].copy()
inc["age"] = [f01.at[i, "age"] for i in inc.customer_id]
inc["grp"] = pd.cut(inc["age"], [17, 27, 39, 100], labels=["18-27", "28-39", "40+"])
bias = inc.groupby("grp", observed=True).apply(
    lambda g: pd.Series({"n": len(g), **{f"sai_lệch_TB_{m}": (g[m] - g["truth"]).mean() for m in METHODS}}),
    include_groups=False)
print("\nannual_income: sai lệch trung bình (điền - thật, triệu/năm) theo nhóm tuổi:")
print(bias.round(1).to_string())

# Kiểm định chéo nội bộ (03) có dự đoán đúng thứ hạng trên dữ liệu thật không?
cv = pd.read_csv(os.path.join(HERE, "internal_cv.csv")).pivot(index="column", columns="method", values="NMAE")
rank_ok = {c: cv.loc[c].idxmin() == e1.loc[c, [f"NMAE_{m}" for m in METHODS]].idxmin().replace("NMAE_", "")
           for c in e1.index}
print("\nCách tốt nhất theo kiểm định nội bộ (03) trùng với cách tốt nhất trên dữ liệu thật (01)?", rank_ok)

# ============================================================================ E2
section("E2. Ô LỖI / NGHI NGỜ ĐÃ SỬA so với file 01")
fx = ch[ch.step_type.isin(["A1", "A5-error", "A5-suspect"])].copy()
fx["truth"] = [f01.at[r.customer_id, r.column] for r in fx.itertuples()]
fx["sai_số_trước"] = (fx["old"] - fx["truth"]).abs()
fx["sai_số_sau"] = (fx["new"] - fx["truth"]).abs()
fx["sai_số_sau_%"] = (fx["sai_số_sau"] / fx["truth"].abs() * 100).round(1)
ev = var[var.cell_type.isin(["error", "suspect"])].set_index(["customer_id", "column"])
for m in METHODS:
    fx[f"nếu_{m}"] = [ev.at[(r.customer_id, r.column), m] if (r.customer_id, r.column) in ev.index else np.nan
                      for r in fx.itertuples()]
print(fx[["customer_id", "column", "step_type", "old", "new", "truth", "sai_số_sau_%"] +
         [f"nếu_{m}" for m in METHODS]].to_string(index=False, float_format=lambda v: f"{v:,.4g}"))

# ============================================================================ E3
section("E3. ĐỐI SOÁT NHIỄU AOV (A3)")
noisy_true = set(log.loc[(log.type == "noise") & (log.column == "avg_order_value"), "customer_id"])
a3 = set(ch.loc[ch.step == "A3", "customer_id"])
tp = a3 & noisy_true
print(f"Ô AOV thực sự có nhiễu (theo log): {len(noisy_true)}; A3 sửa: {len(a3)}; đúng: {len(tp)}")
print(f"  precision = {len(tp)/len(a3):.2f}, recall = {len(tp)/len(noisy_true):.2f}")
idx = [i for i in f01.index if i not in legit]
err = lambda d, ids: ((d.loc[ids, "avg_order_value"] - f01.loc[ids, "avg_order_value"]).abs() / f01.loc[ids, "avg_order_value"])
for name, ids in [("các bản ghi A3 đã sửa", sorted(a3)), ("ô nhiễu A3 KHÔNG bắt được", sorted(noisy_true - a3)),
                  ("toàn bộ (trừ khách mua sỉ, trừ AOV âm)", [i for i in idx if f03.at[i, "avg_order_value"] > 0])]:
    b, a = err(f03, ids), err(f04, ids)
    print(f"  {name:40s} n={len(ids):3d}: sai số tương đối AOV TB {b.mean():.2%} (03) -> {a.mean():.2%} (04); "
          f"trung vị {b.median():.2%} -> {a.median():.2%}")
fp = a3 - noisy_true
if fp:
    print(f"  Bản ghi bị sửa nhầm (không có nhiễu): {sorted(fp)}; sai số sau sửa = "
          f"{err(f04, sorted(fp)).round(4).tolist()}")
print("  Ô nhiễu ở avg_session_minutes, discount_usage_rate (không xử lý): sai số tương đối TB vẫn là",
      {c: f"{((f04.loc[ids, c] - f01.loc[ids, c]).abs() / f01.loc[ids, c].abs().clip(lower=1e-9)).mean():.1%}"
       for c in ["avg_session_minutes", "discount_usage_rate"]
       for ids in [[i for i in log.loc[(log.type == 'noise') & (log.column == c), 'customer_id']
                    if not pd.isna(f04.at[i, c])]]})

# ============================================================================ E4
section("E4. KHÁCH MUA SỈ ĐƯỢC GIỮ NGUYÊN?")
cols = [c for c in f03.columns if c not in ("gender", "preferred_category", "payment_method")]
diff = [(i, c) for i in legit for c in cols if not pd.isna(f03.at[i, c]) and not np.isclose(f03.at[i, c], f04.at[i, c])]
print(f"{len(legit)} khách mua sỉ; số ô quan sát được bị thay đổi: {len(diff)} {diff}")
print("Ô thiếu của họ được điền:", ch[ch.customer_id.isin(legit)][["customer_id", "column", "new", "step_type"]].to_dict("records"))

# ============================================================================ Hình
fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
cols_e = [c for c in e1.index if c != "city_tier"]
w = 0.27
for k, (m, col) in enumerate(zip(METHODS, [GREY, BLUE, ORANGE])):
    ax.bar(np.arange(len(cols_e)) + (k - 1) * w, e1.loc[cols_e, f"NMAE_{m}"], w, color=col, label=m)
ax.set_xticks(range(len(cols_e)), [f"{c}\n(n={e1.at[c,'n']}, chọn {e1.at[c,'chosen']})" for c in cols_e])
ax.set_ylabel("NMAE so với file 01 (MAE / IQR)")
ax.set_title("Sai số điền giá trị thiếu so với dữ liệu sạch")
ax.legend()
ax = axes[1]
gs = bias.index.astype(str)
for k, (m, col) in enumerate(zip(METHODS, [GREY, BLUE, ORANGE])):
    ax.bar(np.arange(len(gs)) + (k - 1) * w, bias[f"sai_lệch_TB_{m}"], w, color=col, label=m)
ax.axhline(0, color="black", lw=.8)
ax.set_xticks(range(len(gs)), [f"{g}\n(n={int(n)})" for g, n in zip(gs, bias["n"])])
ax.set_ylabel("Sai lệch TB (điền - thật), triệu/năm")
ax.set_title("annual_income: sai lệch hệ thống theo nhóm tuổi")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "c_eval_imputation.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n[hình] figures/c_eval_imputation.png")
