"""Builds the model-comparison report from the oof_*.csv files written by train.py.
Output folder: report/  (summary.csv, r2_comparison.png, confusion_matrices.png, scatter_<model>.png)
Confusion matrices: each target is split into Low / Medium / High using tertiles of the TRUE values
(a confusion matrix needs classes, and these are regression targets)."""
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, accuracy_score, f1_score

T = ["ph", "turbidity", "cod", "bod"]
NAMES = {"ph": "pH", "turbidity": "Turbidity (log)", "cod": "COD (log)", "bod": "BOD (log)"}
CLS = ["Low", "Medium", "High"]
os.makedirs("report", exist_ok=True)
rows, cms, data = [], {}, {}
for f in sorted(glob.glob("oof_*.csv")):
    name = os.path.basename(f)[4:-4]; d = pd.read_csv(f); d = d[d.is_synthetic == 0]; data[name] = d   # originals only
    for t in T:
        d2 = d.dropna(subset=[f"true_{t}"]); y, p = d2[f"true_{t}"].values, d2[f"pred_{t}"].values
        cut = np.quantile(y, [1 / 3, 2 / 3]); yc, pc = np.digitize(y, cut), np.digitize(p, cut)
        cms[(name, t)] = confusion_matrix(yc, pc, labels=[0, 1, 2])
        rows.append(dict(model=name, target=NAMES[t], R2=r2_score(y, p), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                         MAE=mean_absolute_error(y, p), Accuracy=accuracy_score(yc, pc), F1_macro=f1_score(yc, pc, average="macro")))
res = pd.DataFrame(rows); models = list(data)
summ = res.groupby("model")[["R2", "RMSE", "MAE", "Accuracy", "F1_macro"]].mean().sort_values("R2", ascending=False)
summ.round(3).to_csv("report/summary.csv"); res.round(3).to_csv("report/per_target.csv", index=False)
print("Mean over the 4 targets:\n", summ.round(3).to_string()); print("\nR2 per target:\n", res.pivot(index="model", columns="target", values="R2").round(3).to_string())

# R2 bar chart
piv = res.pivot(index="model", columns="target", values="R2").loc[summ.index]
ax = piv.plot.bar(figsize=(10, 5), width=.8); ax.axhline(0, color="k", lw=.8); ax.set_ylabel("R² (out-of-fold, original photos)")
ax.set_title("Model comparison: R² per target"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig("report/r2_comparison.png", dpi=150); plt.close()

# Confusion matrices: rows = models, columns = targets
fig, axs = plt.subplots(len(models), 4, figsize=(13, 3.1 * len(models)), squeeze=False)
for i, m in enumerate(summ.index):
    for j, t in enumerate(T):
        a, cm = axs[i][j], cms[(m, t)]; a.imshow(cm, cmap="Blues")
        for r in range(3):
            for c in range(3): a.text(c, r, cm[r, c], ha="center", va="center", color="white" if cm[r, c] > cm.max() / 2 else "black")
        a.set_xticks(range(3), CLS, fontsize=8); a.set_yticks(range(3), CLS, fontsize=8)
        a.set_title(f"{m}\n{NAMES[t]}", fontsize=9)
        if i == len(models) - 1: a.set_xlabel("Predicted")
        if j == 0: a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("report/confusion_matrices.png", dpi=130); plt.close()

# Predicted vs actual scatter, one file per model
for m, d in data.items():
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, t in zip(axs, T):
        d2 = d.dropna(subset=[f"true_{t}"]); y, p = d2[f"true_{t}"], d2[f"pred_{t}"]
        a.scatter(y, p, s=14, alpha=.7); lo, hi = min(y.min(), p.min()), max(y.max(), p.max()); a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_title(f"{NAMES[t]}  R²={r2_score(y, p):.2f}", fontsize=10); a.set_xlabel("Actual"); a.set_ylabel("Predicted")
    fig.suptitle(m); plt.tight_layout(); plt.savefig(f"report/scatter_{m}.png", dpi=120); plt.close()
print("\nSaved to report/")
