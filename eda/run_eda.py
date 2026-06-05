import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- config ---
DATA_PATH = "WineQT.csv"
OUT_DIR   = Path("data_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
QUALITY_PALETTE = sns.color_palette("coolwarm", n_colors=6)  # quality 3–8

# --- load ---
df       = pd.read_csv(DATA_PATH).drop(columns=["Id"])
features = [c for c in df.columns if c != "quality"]
corr     = df.corr()

# --- 1. Quality distribution (bar graph) ---
quality_counts = df["quality"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(quality_counts.index, quality_counts.values,
       color=QUALITY_PALETTE, edgecolor="white")
ax.set(title="Wine Quality Distribution", xlabel="Quality Score", ylabel="Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_quality_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

#--- 2. Feature distributions ---
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")

for ax, col in zip(axes.flat, features):
    ax.hist(df[col], bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(col, fontsize=9)
    ax.set_ylabel("Count", fontsize=8)

for ax in axes.flat[len(features):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 4. Quality correlations ---
qual_corr = corr["quality"].drop("quality").sort_values()
colors    = ["#d62728" if v < 0 else "#2ca02c" for v in qual_corr]

fig, ax = plt.subplots(figsize=(8, 5))
qual_corr.plot(kind="barh", color=colors, ax=ax, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlations with Quality", fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson r")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_quality_correlations.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 5. Box plots (key features by quality) ---
key_features  = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
quality_order = sorted(df["quality"].unique())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Key Features by Quality Score", fontsize=14, fontweight="bold")

for ax, feat in zip(axes.flat, key_features):
    groups = [df.loc[df["quality"] == q, feat] for q in quality_order]
    ax.boxplot(groups, tick_labels=quality_order,
               patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6),
               medianprops=dict(color="black", linewidth=1.5))
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel("Quality")

plt.tight_layout()
plt.savefig(OUT_DIR / "05_boxplots_by_quality.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 6. Scatter(alcohol vs volatile acidity) ---
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df["alcohol"], df["volatile acidity"],
                     c=df["quality"], cmap="coolwarm", alpha=0.6,
                     edgecolors="white", linewidth=0.3, s=40)
plt.colorbar(scatter, ax=ax, label="Quality")
ax.set(xlabel="Alcohol", ylabel="Volatile Acidity",
       title="Alcohol vs Volatile Acidity (coloured by Quality)")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_scatter_alcohol_vs_acidity.png", dpi=150, bbox_inches="tight")
plt.close()

print("Complete.")
