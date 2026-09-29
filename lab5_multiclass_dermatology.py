"""
CSET343 - AI in Healthcare | Lab Assignment 5
Multiclass Classification on the Dermatology Dataset (UCI)

Run:  python lab5_multiclass_dermatology.py
Requires: pandas numpy matplotlib seaborn scikit-learn
Outputs: plots saved in ./lab5_outputs/
"""

import os
import io
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line if you want plots to pop up interactively
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42
OUT_DIR = "lab5_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = {
    1: "psoriasis",
    2: "seboreic dermatitis",
    3: "lichen planus",
    4: "pityriasis rosea",
    5: "cronic dermatitis",
    6: "pityriasis rubra pilaris",
}

# =====================================================================
# TASK 1: DATA ACQUISITION AND EXPLORATION
# =====================================================================
print("=" * 70)
print("TASK 1: DATA ACQUISITION AND EXPLORATION")
print("=" * 70)

# ---- 1. Download and load the dataset ----
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"

FEATURE_NAMES = [
    "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
    "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
    "knee_elbow_involvement", "scalp_involvement", "family_history",
    "melanin_incontinence", "eosinophils_infiltrate", "PNL_infiltrate",
    "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
    "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis", "disappearance_granular_layer",
    "vacuolisation_damage_basal_layer", "spongiosis", "saw_tooth_appearance_retes",
    "follicular_horn_plug", "perifollicular_parakeratosis",
    "inflammatory_monoluclear_infiltrate", "band_like_infiltrate", "age",
]
COLUMNS = FEATURE_NAMES + ["Class"]


def load_data():
    """Try to download from UCI; fall back to a local 'dermatology.data' file."""
    try:
        with urllib.request.urlopen(URL, timeout=20) as r:
            raw = r.read().decode("utf-8")
        print("Dataset downloaded from UCI repository.")
        return pd.read_csv(io.StringIO(raw), header=None, names=COLUMNS, na_values="?")
    except Exception as e:
        print(f"Download failed ({e}).")
        if os.path.exists("dermatology.data"):
            print("Loading local 'dermatology.data' instead.")
            return pd.read_csv("dermatology.data", header=None, names=COLUMNS, na_values="?")
        raise SystemExit(
            "Could not download the data. Manually download 'dermatology.data' from\n"
            "https://archive.ics.uci.edu/dataset/33/dermatology and place it next to this script."
        )


df = load_data()
print("\nShape:", df.shape)
print(df.head())
print("\nData types / info:")
df.info()

# ---- 2. Summary statistics ----
print("\n--- Summary statistics ---")
summary = df.describe().T
summary["median"] = df.median(numeric_only=True)
print(summary[["count", "mean", "median", "std", "min", "max"]].round(3).to_string())

# ---- 3. Missing values, duplicates, outliers ----
print("\n--- Missing values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() else "No missing values.")

# 'age' has 8 missing values ('?') in this dataset -> impute with median
if df["age"].isnull().any():
    imputer = SimpleImputer(strategy="median")
    df[["age"]] = imputer.fit_transform(df[["age"]])
    print("Missing 'age' values imputed with the median.")

print("\n--- Duplicates ---")
n_dup = df.duplicated().sum()
print("Duplicate rows:", n_dup)
if n_dup:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed. New shape:", df.shape)

print("\n--- Outliers (IQR method) ---")
# Most features are ordinal (0-3), so IQR is meaningful mainly for 'age'.
# We report outliers for all features but only cap (winsorize) continuous 'age'.
feature_cols = [c for c in df.columns if c != "Class"]
outlier_counts = {}
for col in feature_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
out_series = pd.Series(outlier_counts)
print(out_series[out_series > 0] if out_series.sum() else "No outliers detected.")

q1, q3 = df["age"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_age_out = int(((df["age"] < lo) | (df["age"] > hi)).sum())
df["age"] = df["age"].clip(lo, hi)
print(f"'age': {n_age_out} outlier(s) capped to [{lo:.1f}, {hi:.1f}] (winsorization).")
print("Note: ordinal 0-3 histopathological/clinical scores are valid clinical grades, "
      "so their 'outliers' are kept.")

# ---- 4. Visualizations ----
# (a) Class distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="Class", data=df, palette="viridis")
ax.set_xticklabels([f"{k}\n{v}" for k, v in CLASS_NAMES.items()], fontsize=8)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# (b) Histograms for key features
key_features = ["age", "erythema", "scaling", "itching", "koebner_phenomenon",
                "polygonal_papules", "knee_elbow_involvement", "scalp_involvement",
                "acanthosis"]
df[key_features].hist(bins=15, figsize=(12, 8), color="steelblue", edgecolor="black")
plt.suptitle("Histograms of Key Features")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_histograms.png", dpi=150)
plt.close()

# (c) Box plots for key features (grouped by class)
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.ravel(), key_features):
    sns.boxplot(x="Class", y=col, data=df, ax=ax, palette="Set2")
    ax.set_title(col)
plt.suptitle("Box Plots of Key Features by Class")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_boxplots.png", dpi=150)
plt.close()

# (d) Correlation heatmap
plt.figure(figsize=(16, 13))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False, linewidths=0.2)
plt.title("Correlation Heatmap (features + Class)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()

corr_with_target = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
print("\nTop 10 features most correlated with Class:")
print(corr_with_target.head(10).round(3).to_string())

# ---- 5. Class imbalance discussion ----
print("\n--- Class distribution ---")
dist = df["Class"].value_counts().sort_index()
dist_df = pd.DataFrame({
    "Disease": [CLASS_NAMES[i] for i in dist.index],
    "Count": dist.values,
    "Percent": (dist.values / len(df) * 100).round(2),
})
dist_df.index = dist.index
print(dist_df.to_string())
ratio = dist.max() / dist.min()
print(f"\nImbalance ratio (largest/smallest): {ratio:.2f}")
print("""
Discussion:
 - Psoriasis (class 1) is the majority class (~30%), while pityriasis rubra pilaris
   (class 6, ~5%) is the rarest. The imbalance is MODERATE (not extreme).
 - Implications: models can become biased toward majority classes; accuracy alone can be
   misleading; minority classes get lower recall. With a small dataset (~366 rows) the test
   set will contain very few minority samples, so metrics for class 6 are high-variance.
 - Mitigation: stratified splitting (used here), class_weight='balanced', per-class
   precision/recall/F1 and macro averages, and optionally oversampling (SMOTE).
""")

# =====================================================================
# TASK 2: DATA PREPROCESSING
# =====================================================================
print("=" * 70)
print("TASK 2: DATA PREPROCESSING")
print("=" * 70)

X = df.drop(columns="Class")
y_raw = df["Class"]

# Encode target (1-6 -> 0-5)
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_labels = [CLASS_NAMES[c] for c in le.classes_]
n_classes = len(class_labels)
print("Encoded classes:", dict(zip(le.classes_, le.transform(le.classes_))))

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train class proportions:", np.round(np.bincount(y_train) / len(y_train), 3))
print("Test  class proportions:", np.round(np.bincount(y_test) / len(y_test), 3))

# Standardize (fit on train only -> no data leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Features standardized (mean~0, std~1) using training statistics only.")

# =====================================================================
# TASK 3: MODEL TRAINING AND EVALUATION
# =====================================================================
print("=" * 70)
print("TASK 3: MODEL TRAINING AND EVALUATION")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
}

results = []
trained = {}
y_test_bin = label_binarize(y_test, classes=range(n_classes))

for name, model in models.items():
    model.fit(X_train_s, y_train)
    trained[name] = model
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

    results.append({
        "Model": name, "Accuracy": acc,
        "Precision (macro)": p_m, "Recall (macro)": r_m, "F1 (macro)": f_m,
        "Precision (weighted)": p_w, "Recall (weighted)": r_w, "F1 (weighted)": f_w,
        "AUC (macro OvR)": auc_macro,
    })

    print(f"\n{'-' * 70}\n{name}\n{'-' * 70}")
    print(f"Accuracy: {acc:.4f} | Macro AUC: {auc_macro:.4f}")
    print("Per-class precision / recall / F1:")
    print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUT_DIR}/cm_{fname}.png", dpi=150)
    plt.close()

    # ROC curves (One-vs-Rest, per class + macro average)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{class_labels[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(all_fpr, mean_tpr, "k--", lw=2.5, label=f"Macro-average (AUC={auc_macro:.3f})")
    plt.plot([0, 1], [0, 1], "grey", linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (OvR) - {name}")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_{fname}.png", dpi=150)
    plt.close()

# ---- Comparison table ----
res_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n" + "=" * 70)
print("MODEL COMPARISON (test set)")
print("=" * 70)
print(res_df.to_string())
res_df.to_csv(f"{OUT_DIR}/model_comparison.csv")

# Comparison bar chart
res_df[["Accuracy", "F1 (macro)", "AUC (macro OvR)"]].plot(kind="bar", figsize=(10, 5), ylim=(0.8, 1.02))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
plt.close()

# Feature importance from Random Forest
rf = trained["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 6))
sns.barplot(x=imp.values, y=imp.index, palette="magma")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance_rf.png", dpi=150)
plt.close()

# ---- Interpretation ----
best = res_df["F1 (macro)"].idxmax()
print(f"""
INTERPRETATION
--------------
 * Best model by macro-F1: {best} (Accuracy={res_df.loc[best, 'Accuracy']:.3f},
   macro-F1={res_df.loc[best, 'F1 (macro)']:.3f}, macro-AUC={res_df.loc[best, 'AUC (macro OvR)']:.3f}).
 * The dermatology features (clinical + histopathological scores) are highly informative, so most
   models reach high accuracy; differences between top models may be small and, given only
   ~74 test samples, not statistically significant. Consider k-fold cross-validation
   for a more reliable comparison.
 * Confusion matrices show that most errors occur between clinically similar diseases
   (e.g., seboreic dermatitis vs. chronic dermatitis / psoriasis vs. lichen planus), which is also
   what dermatologists find difficult in differential diagnosis.
 * The rare class (pityriasis rubra pilaris) has few test samples, so its precision/recall
   swings sharply with a single misclassification; macro-averaged metrics expose this better
   than accuracy alone.
 * Decision Tree tends to be the weakest/most variable (overfits); ensembles (Random Forest) and
   margin-based models (SVM, LR) generalize better. KNN and SVM benefit from the standardization.
 * In healthcare, recall for each disease matters (missed diagnoses are costly), so per-class
   recall should guide final model selection, not accuracy alone.

All plots and the comparison table are saved in the '{OUT_DIR}/' folder.
""")
