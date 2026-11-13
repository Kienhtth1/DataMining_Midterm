"""
Câu a - Đồ thị minh chứng quá trình sinh dữ liệu.
Chỉ đọc các file trong data/, lưu PNG (dpi 200) vào figures/.
Ngoại lai / khách mua sỉ được xác định từ injection_log (cột type), không hard-code chỉ số dòng.

  fig_noise.png   - phân phối trước (01) / sau nhiễu (02) trên các bản ghi không phải ngoại lai
  fig_outlier.png - orders_12m vs avg_order_value: khách mua sỉ tách biệt khỏi 4 phân khúc
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
DPI = 200

f01 = pd.read_csv(os.path.join(DATA, "01_clean_ground_truth.csv"))
f02 = pd.read_csv(os.path.join(DATA, "02_noisy_outliers.csv"))
log = pd.read_csv(os.path.join(DATA, "injection_log.csv"))

# Hai file phải cùng thứ tự bản ghi để so sánh theo dòng
assert (f01["customer_id"] == f02["customer_id"]).all(), "01 và 02 lệch thứ tự customer_id"

# ---------------------------------------------------------------- Xác định nhóm bản ghi từ log
outlier_ids = set(log.loc[log["type"].str.startswith("outlier"), "customer_id"])
error_ids = set(log.loc[log["type"] == "outlier_error", "customer_id"])
legit_ids = set(log.loc[log["type"] == "outlier_legit", "customer_id"])
assert not (error_ids & legit_ids), "Một bản ghi vừa là lỗi vừa là mua sỉ"

# ================================================================ Đồ thị 1 - Nhiễu
C_BEFORE, C_AFTER = "#2a6f97", "#e07a5f"
mask_normal = ~f01["customer_id"].isin(outlier_ids)
before = f01.loc[mask_normal]
after = f02.loc[mask_normal]

NOISE_COLS = [
    ("avg_order_value", "log10(avg_order_value) [VND]", np.log10),
    ("avg_session_minutes", "avg_session_minutes [phút/tuần]", None),
    ("discount_usage_rate", "discount_usage_rate", None),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
stats = []
for ax, (col, xlabel, tf) in zip(axes, NOISE_COLS):
    b = before[col].astype(float).to_numpy()
    a = after[col].astype(float).to_numpy()
    if tf is not None:
        b, a = tf(b), tf(a)
    edges = np.histogram_bin_edges(np.concatenate([b, a]), bins=20)   # chung 20 bin
    ax.hist(b, bins=edges, color=C_BEFORE, alpha=0.6, label="Trước (sạch)", edgecolor="white", linewidth=0.5)
    ax.hist(a, bins=edges, color=C_AFTER, alpha=0.6, label="Sau nhiễu", edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Số bản ghi")
    n_noisy = int(((log["type"] == "noise") & (log["column"] == col)).sum())
    ax.set_title(f"{col}\n({n_noisy}/{len(b)} ô có nhiễu)", fontsize=11)
    ax.text(0.98, 0.97,
            f"Trước: {b.mean():.3f} ± {b.std(ddof=1):.3f}\nSau:    {a.mean():.3f} ± {a.std(ddof=1):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)          # chừa khoảng trống cho hộp thống kê
    ax.spines[["top", "right"]].set_visible(False)
    stats.append((col, "log10" if tf is not None else "", len(b), b, a))

fig.suptitle(f"Phân phối trước/sau khi thêm nhiễu - {mask_normal.sum()} bản ghi không phải ngoại lai (file 01 vs 02)",
             fontsize=12, y=0.99)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.93))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(FIG, "fig_noise.png"), dpi=DPI)
plt.close(fig)

# ================================================================ Đồ thị 2 - Khách mua sỉ
d2 = f02.merge(f01[["customer_id", "true_segment"]], on="customer_id", how="left")
d2 = d2.loc[~d2["customer_id"].isin(error_ids)].copy()          # bỏ 8 ngoại lai do lỗi
d2["aov_m"] = d2["avg_order_value"] / 1e6
assert (d2["aov_m"] > 0).all() and (d2["orders_12m"] > 0).all(), "Giá trị ≤ 0 không vẽ được trên thang log"

is_legit = d2["customer_id"].isin(legit_ids)
normal = d2.loc[~is_legit]
legit = d2.loc[is_legit]

SEG_ORDER = ["VIP", "Bargain", "Occasional", "ChurnRisk"]
SEG_COLOR = {"VIP": "#7b2cbf", "Bargain": "#2a9d8f", "Occasional": "#e9c46a", "ChurnRisk": "#adb5bd"}

fig, ax = plt.subplots(figsize=(8.5, 6.2))
for seg in SEG_ORDER:
    s = normal.loc[normal["true_segment"] == seg]
    ax.scatter(s["orders_12m"], s["aov_m"], s=28, alpha=0.8, color=SEG_COLOR[seg],
               edgecolor="white", linewidth=0.4, label=f"{seg} (n={len(s)})")
ax.scatter(legit["orders_12m"], legit["aov_m"], marker="*", s=230, color="black",
           edgecolor="white", linewidth=0.6, label=f"Khách mua sỉ (n={len(legit)})", zorder=5)

ax.set_xscale("log")
ax.set_yscale("log")
plain = FuncFormatter(lambda v, _: f"{v:g}")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(plain)
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_xlabel("orders_12m (số đơn / 12 tháng, thang log)")
ax.set_ylabel("avg_order_value (triệu VND, thang log)")
ax.set_title(f"Khách mua sỉ tách biệt khỏi 4 phân khúc - file 02, {len(d2)} bản ghi\n"
             f"(đã bỏ {len(error_ids)} ngoại lai do lỗi)", fontsize=11)
ax.grid(True, which="major", color="0.9", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=9, frameon=True)   # góc trên-trái trống, không che điểm
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_outlier.png"), dpi=DPI)
plt.close(fig)

# ================================================================ In kết quả
print("== Số bản ghi dùng cho mỗi đồ thị ==")
print(f"fig_noise.png  : {mask_normal.sum()} bản ghi (200 - {len(outlier_ids)} ngoại lai)")
print(f"fig_outlier.png: {len(d2)} bản ghi = {len(normal)} thường + {len(legit)} mua sỉ "
      f"(đã bỏ {len(error_ids)} lỗi)")
print("   theo phân khúc:", normal["true_segment"].value_counts().reindex(SEG_ORDER).to_dict())
print("\n== Trung bình ± độ lệch chuẩn (ddof=1), trước / sau nhiễu ==")
for col, tf, n, b, a in stats:
    name = f"{tf}({col})" if tf else col
    print(f"{name:28s} n={n}  trước {b.mean():.4f} ± {b.std(ddof=1):.4f}   "
          f"sau {a.mean():.4f} ± {a.std(ddof=1):.4f}   Δmean={a.mean()-b.mean():+.4f}")
print(f"\nĐã lưu: {os.path.join('figures', 'fig_noise.png')}, {os.path.join('figures', 'fig_outlier.png')}")
