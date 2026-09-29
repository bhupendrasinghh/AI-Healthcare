"""
CSET343 - AI in Healthcare | Lab Assignment 4
Logistic Regression on the Breast Cancer Wisconsin (Diagnostic) Dataset

Run:  python lab4_logistic_regression_breast_cancer.py
Requires: pandas numpy matplotlib seaborn scikit-learn
Outputs: plots saved in ./lab4_outputs/
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line to display plots interactively
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_curve, roc_auc_score)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42
OUT_DIR = "lab4_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# TASK 1: DATA ACQUISITION AND EXPLORATION
# =====================================================================
print("=" * 70)
print("TASK 1: DATA ACQUISITION AND EXPLORATION")
print("=" * 70)

# ---- 1. Download and load the dataset ----
# Loaded from scikit-learn (identical to the UCI WDBC dataset, no internet needed).
# Alternative: pd.read_csv("data.csv") if you downloaded the Kaggle version.
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

# In sklearn: target 0 = malignant, 1 = benign.
# We define 'Outcome' so that 1 = Malignant (the positive/disease class) and 0 = Benign.
df["Outcome"] = (df["target"] == 0).astype(int)
df = df.drop(columns="target")

print("Shape:", df.shape)
print(df.head())
print("\nInfo:")
df.info()

# ---- 2. Summary statistics ----
print("\n--- Summary statistics ---")
summary = df.describe().T
summary["median"] = df.median()
print(summary[["count", "mean", "median", "std", "min", "max"]].round(3).to_string())

# ---- 3. Missing values, duplicates, outliers ----
print("\n--- Missing values ---")
n_missing = df.isnull().sum().sum()
print("Total missing values:", n_missing)
if n_missing:
    df = df.fillna(df.median())
    print("Missing values imputed with column medians.")

print("\n--- Duplicates ---")
n_dup = df.duplicated().sum()
print("Duplicate rows:", n_dup)
if n_dup:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed. New shape:", df.shape)

print("\n--- Outliers (IQR method) ---")
feature_cols = [c for c in df.columns if c != "Outcome"]
outlier_counts = {}
for col in feature_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
out_series = pd.Series(outlier_counts).sort_values(ascending=False)
print("Outlier counts per feature (top 10):")
print(out_series.head(10).to_string())
print(f"Total flagged values: {out_series.sum()}")

# Handling: winsorize (cap) at the IQR fences instead of deleting rows.
# In medical data, extreme values are often genuine (large tumours), and the dataset
# is small (569 rows), so capping keeps every patient while limiting outlier influence.
for col in feature_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
print("Outliers capped (winsorized) at 1.5*IQR fences.")

# ---- 4. Visualizations ----
# (a) Class distribution
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Outcome", data=df, palette=["#4C9F70", "#D9534F"])
ax.set_xticklabels(["Benign (0)", "Malignant (1)"])
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# (b) Histograms of key features by class
key_features = ["mean radius", "mean texture", "mean perimeter", "mean area",
                "mean concavity", "mean concave points"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), key_features):
    sns.histplot(data=df, x=col, hue="Outcome", bins=30, kde=True, ax=ax,
                 palette=["#4C9F70", "#D9534F"])
    ax.set_title(col)
plt.suptitle("Histograms of Key Features (0=Benign, 1=Malignant)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_histograms.png", dpi=150)
plt.close()

# (c) Box plots of key features by class
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), key_features):
    sns.boxplot(x="Outcome", y=col, data=df, ax=ax, palette=["#4C9F70", "#D9534F"])
    ax.set_title(col)
plt.suptitle("Box Plots of Key Features by Class (after outlier capping)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_boxplots.png", dpi=150)
plt.close()

# (d) Correlation heatmap
plt.figure(figsize=(16, 13))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, linewidths=0.2)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()

corr_target = df.corr()["Outcome"].drop("Outcome").abs().sort_values(ascending=False)
print("\nTop 10 features correlated with Outcome:")
print(corr_target.head(10).round(3).to_string())
print("\nNote: radius, perimeter and area (and their 'mean/worst' versions) are highly "
      "correlated with each other -> multicollinearity, which regularization helps handle.")

# ---- 5. Class imbalance discussion ----
print("\n--- Class distribution ---")
counts = df["Outcome"].value_counts().sort_index()
pct = (counts / len(df) * 100).round(2)
print(pd.DataFrame({"Class": ["Benign (0)", "Malignant (1)"], "Count": counts.values,
                    "Percent": pct.values}).to_string(index=False))
print(f"Imbalance ratio (majority/minority): {counts.max() / counts.min():.2f}")
print("""
Discussion:
 - Benign ~63% vs Malignant ~37%: a mild imbalance (ratio ~1.7), not severe.
 - A model that always predicts 'benign' would already reach ~63% accuracy, so accuracy alone
   is not a sufficient metric -- precision, recall, F1 and AUC must be examined.
 - Mild imbalance can bias the decision boundary toward the majority class and reduce recall on
   malignant cases, which is the costlier error in diagnosis.
 - Mitigations: stratified splitting (used here), class_weight='balanced', threshold tuning,
   or resampling (e.g. SMOTE) if the imbalance were stronger.
""")

# =====================================================================
# TASK 2: DATA PREPROCESSING
# =====================================================================
print("=" * 70)
print("TASK 2: DATA PREPROCESSING")
print("=" * 70)

X = df.drop(columns="Outcome")
y = df["Outcome"]  # already binary: 1 = Malignant, 0 = Benign
print("Target encoded as binary: Malignant=1, Benign=0")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train class proportions:", y_train.value_counts(normalize=True).sort_index().round(3).to_dict())
print("Test  class proportions:", y_test.value_counts(normalize=True).sort_index().round(3).to_dict())

# Standardize (fit on training data only to avoid data leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Features standardized using training-set mean and std.")

# =====================================================================
# TASK 3: MODEL TRAINING AND EVALUATION
# =====================================================================
print("=" * 70)
print("TASK 3: MODEL TRAINING AND EVALUATION")
print("=" * 70)

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

print("\nPrecision / Recall / F1 for BOTH classes:")
print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"], digits=4))

for label, name in [(0, "Benign"), (1, "Malignant")]:
    p = precision_score(y_test, y_pred, pos_label=label)
    r = recall_score(y_test, y_pred, pos_label=label)
    f = f1_score(y_test, y_pred, pos_label=label)
    print(f"{name:<10} -> Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix:\n{cm}")
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Specificity (TNR): {tn / (tn + fp):.4f}   Sensitivity (TPR): {tp / (tp + fn):.4f}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_confusion_matrix.png", dpi=150)
plt.close()

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, color="darkorange", label=f"Logistic Regression (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_roc_curve.png", dpi=150)
plt.close()

# ---- Model interpretation: coefficients ----
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("\nTop 5 features pushing toward MALIGNANT (largest positive coefficients):")
print(coef.tail(5)[::-1].round(3).to_string())
print("\nTop 5 features pushing toward BENIGN (largest negative coefficients):")
print(coef.head(5).round(3).to_string())

top = pd.concat([coef.head(8), coef.tail(8)])
plt.figure(figsize=(8, 6))
top.plot(kind="barh", color=["#4C9F70" if v < 0 else "#D9534F" for v in top])
plt.title("Logistic Regression Coefficients (standardized features)")
plt.xlabel("Coefficient (log-odds per 1 std increase)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_coefficients.png", dpi=150)
plt.close()

# ---- Bonus: Regularization (L1 vs L2, effect of C) ----
print("\n" + "=" * 70)
print("BONUS: REGULARIZATION EXPLORATION")
print("=" * 70)
rows = []
for penalty in ["l1", "l2"]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        m = LogisticRegression(penalty=penalty, C=C, solver="liblinear",
                               max_iter=1000, random_state=RANDOM_STATE)
        m.fit(X_train_s, y_train)
        pr = m.predict(X_test_s)
        pb = m.predict_proba(X_test_s)[:, 1]
        rows.append({
            "Penalty": penalty.upper(), "C": C,
            "Accuracy": accuracy_score(y_test, pr),
            "Recall (Malignant)": recall_score(y_test, pr),
            "F1 (Malignant)": f1_score(y_test, pr),
            "AUC": roc_auc_score(y_test, pb),
            "Non-zero coefs": int((m.coef_ != 0).sum()),
        })
reg_df = pd.DataFrame(rows).round(4)
print(reg_df.to_string(index=False))
reg_df.to_csv(f"{OUT_DIR}/regularization_results.csv", index=False)

plt.figure(figsize=(7, 4.5))
for penalty in ["L1", "L2"]:
    sub = reg_df[reg_df["Penalty"] == penalty]
    plt.semilogx(sub["C"], sub["Accuracy"], marker="o", label=f"{penalty} accuracy")
plt.xlabel("C (inverse regularization strength)")
plt.ylabel("Test accuracy")
plt.title("Effect of Regularization Strength")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_regularization.png", dpi=150)
plt.close()

# ---- Interpretation ----
print(f"""
INTERPRETATION
--------------
 * Accuracy = {acc:.3f}, AUC = {roc_auc:.3f}: the model separates benign and malignant tumours very well,
   far above the ~63% baseline of always predicting the majority class.
 * Confusion matrix: FN={fn} (malignant tumours missed) and FP={fp} (benign flagged as malignant).
   In cancer diagnosis a false negative is the more dangerous error -- a missed cancer delays
   treatment -- so RECALL (sensitivity) on the malignant class is the key metric. A false positive
   causes anxiety and extra tests (biopsy), which is costly but less harmful.
 * Precision on the malignant class tells us how many positive alerts are true cancers; high
   precision limits unnecessary biopsies. F1 balances precision and recall.
 * AUC is threshold-independent; the threshold can be lowered (e.g. 0.3-0.4) to push recall
   even higher at the cost of more false positives -- a sensible choice for screening.
 * Coefficients (printed above) show which nuclei measurements (e.g. texture, size/area errors,
   concave points, symmetry) raise the log-odds of malignancy and which lower it. Because the
   features are standardized, coefficient sizes are comparable, which makes logistic regression
   easy to explain to clinicians. Note that correlated features (radius/perimeter/area) share
   weight, so individual coefficients should be read with caution.
 * Regularization: strong penalties (small C) shrink coefficients and L1 zeroes out many
   features (built-in feature selection); moderate C values usually give the best trade-off,
   which helps against the multicollinearity seen in the heatmap.
 * Caveats: the test set has only ~114 patients, so metrics carry uncertainty; use k-fold
   cross-validation and external validation before any clinical deployment. The model
   should support, not replace, a clinician's judgement.

All plots and tables are saved in the '{OUT_DIR}/' folder.
""")
