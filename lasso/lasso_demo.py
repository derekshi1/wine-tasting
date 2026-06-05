import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

#--- config ----
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("WineQT.csv")
OUT_DIR = Path("lasso_output")
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# ---- load ----
df = pd.read_csv(CSV_PATH)
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

X = df.drop(columns=["quality"])
y = df["quality"].astype(float)
feat_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---- fit ----
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=cv, n_alphas=200, max_iter=20000, random_state=RANDOM_STATE)),
])
pipe.fit(X_train, y_train)
lasso = pipe.named_steps["lasso"]
best_alpha = lasso.alpha_

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

# ---- report ----
print("=" * 60)
print("LASSO REGRESSION: Wine Quality")
print("=" * 60)
print(f"n samples: {len(df)}  |  n features: {X.shape[1]}")
print(f"Train/test: {len(X_train)}/{len(X_test)}  |  CV: 5-fold")
print(f"Best alpha: {best_alpha:.6f}")
print()
print("Performance")
print(f"  Train  R^2 = {r2_score(y_train, y_pred_train):.4f}   "
      f"RMSE = {mean_squared_error(y_train, y_pred_train)**0.5:.4f}   "
      f"MAE = {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"  Test   R^2 = {r2_score(y_test, y_pred_test):.4f}   "
      f"RMSE = {mean_squared_error(y_test, y_pred_test)**0.5:.4f}   "
      f"MAE = {mean_absolute_error(y_test, y_pred_test):.4f}")
print()

coefs = (
    pd.DataFrame({"feature": feat_names, "coef_standardized": lasso.coef_})
    .assign(abs_coef=lambda d: d["coef_standardized"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns="abs_coef")
)
print("Coefficients (on standardized features)")
print(coefs.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
print()
print(f"Features zeroed out: {int((lasso.coef_ == 0).sum())} / {X.shape[1]}")

# ---- save ----
coefs.to_csv(OUT_DIR / "lasso_coefficients.csv", index=False)

test_out = X_test.copy()
test_out["quality_actual"] = y_test.values
test_out["quality_predicted"] = y_pred_test
test_out["residual"] = test_out["quality_actual"] - test_out["quality_predicted"]
test_out.to_csv(OUT_DIR / "lasso_test_predictions.csv", index=False)

# ---- plots ----
alphas = lasso.alphas_
mse_path = lasso.mse_path_.mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(np.log10(alphas), mse_path)
axes[0].axvline(np.log10(best_alpha), color="red", ls="--",
                label=f"best a = {best_alpha:.4f}")
axes[0].set_xlabel("log10(alpha)")
axes[0].set_ylabel("Mean CV MSE")
axes[0].set_title("LassoCV: CV error vs alpha")
axes[0].legend(); axes[0].grid(alpha=0.3)

vals = coefs["coef_standardized"].values
colors = ["#888" if v == 0 else ("#2a7" if v > 0 else "#c33") for v in vals]
axes[1].barh(range(len(coefs)), vals, color=colors)
axes[1].set_yticks(range(len(coefs)))
axes[1].set_yticklabels(coefs["feature"].tolist())
axes[1].invert_yaxis()
axes[1].axvline(0, color="black", lw=0.6)
axes[1].set_xlabel("Standardized coefficient")
axes[1].set_title("Lasso coefficients")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.show()