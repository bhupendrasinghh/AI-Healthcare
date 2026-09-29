"""
CSET343 - AI in Healthcare | Lab Assignment 6
CNN + handcrafted features (HOG, LBP, GLCM) on Chest X-Ray Images (Normal vs Pneumonia)

Dataset (Kaggle): "Chest X-Ray Images (Pneumonia)" by Paul Mooney (~5,863 images)
    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Extract it so the folder looks like:
    chest_xray/train/NORMAL, chest_xray/train/PNEUMONIA, chest_xray/val/..., chest_xray/test/...
Set DATA_DIR below to that 'chest_xray' folder.

Install:  pip install tensorflow scikit-learn scikit-image pillow matplotlib pandas numpy
Run:      python lab6_chest_xray_cnn.py
"""

import os
import random
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
DATA_DIR = "chest_xray"          # <-- change to your dataset path
OUT_DIR = "lab6_outputs"         # plots and tables are saved here
IMG_SIZE = 128
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # label 0 / label 1
SEED = 42
CNN_CV_FOLDS = 3
CNN_CV_EPOCHS = 8
CNN_FINAL_EPOCHS = 25

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------
# Task 1: Data acquisition and exploration
# ---------------------------------------------------------------
def collect_paths(data_dir):
    """Pool every image from train/val/test folders; label comes from the class folder."""
    paths, labels = [], []
    for p in Path(data_dir).rglob("*"):
        if p.suffix.lower() in (".jpeg", ".jpg", ".png") and p.parent.name.upper() in CLASS_NAMES:
            paths.append(str(p))
            labels.append(CLASS_NAMES.index(p.parent.name.upper()))
    if not paths:
        raise FileNotFoundError(f"No images found under '{data_dir}'. Check DATA_DIR.")
    return np.array(paths), np.array(labels)


def show_samples(paths, labels, n=6):
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4))
    for row, cls in enumerate(CLASS_NAMES):
        idx = np.where(labels == row)[0]
        for col, i in enumerate(random.sample(list(idx), n)):
            axes[row, col].imshow(Image.open(paths[i]).convert("L"), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(cls, loc="left", fontsize=11, fontweight="bold")
    plt.suptitle("Random samples per class")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_sample_images.png", dpi=120)
    plt.show()


def report_balance(labels):
    counts = pd.Series(labels).map(dict(enumerate(CLASS_NAMES))).value_counts()
    print("\nImages per class:\n", counts)
    print(f"Total images: {counts.sum()}")
    print(f"Imbalance ratio (majority/minority): {counts.max() / counts.min():.2f}")
    counts.plot(kind="bar", color=["#4C72B0", "#DD8452"], rot=0)
    plt.title("Class balance")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_class_balance.png", dpi=120)
    plt.show()


# ---------------------------------------------------------------
# Task 2: Preprocessing and feature extraction
# ---------------------------------------------------------------
def load_image(path):
    """Grayscale + resize to IMG_SIZE x IMG_SIZE, returns uint8 array."""
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def hog_features(img):
    return hog(img, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys")


def lbp_features(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist


def glcm_features(img, levels=32):
    q = (img // (256 // levels)).astype(np.uint8)          # quantise to 32 gray levels
    glcm = graycomatrix(q, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props])


def extract_all(images):
    feats = []
    for k, img in enumerate(images):
        feats.append(np.concatenate([hog_features(img), lbp_features(img), glcm_features(img)]))
        if (k + 1) % 1000 == 0:
            print(f"  features extracted: {k + 1}/{len(images)}")
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------
# Evaluation helpers (Task 3)
# ---------------------------------------------------------------
def compute_metrics(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_prob),
    }


def plot_confusion(y_true, y_prob, title, fname):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=120)
    plt.show()
    return cm


# ---------------------------------------------------------------
# CNN
# ---------------------------------------------------------------
def build_cnn(filters=16, dropout=0.4, lr=1e-3, dense_units=64):
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.RandomRotation(0.05),                 # light augmentation (train time only)
        layers.RandomZoom(0.1),
        layers.Conv2D(filters, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(filters * 2, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(filters * 4, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def cnn_cross_validation(X_tr, y_tr, class_weight):
    """K-fold CV over a small hyperparameter grid; returns best params."""
    grid = {"filters": [16, 32], "dropout": [0.3, 0.5], "lr": [1e-3, 3e-4]}
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    skf = StratifiedKFold(n_splits=CNN_CV_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for params in combos:
        fold_auc = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            keras.backend.clear_session()
            m = build_cnn(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx], epochs=CNN_CV_EPOCHS, batch_size=32,
                  class_weight=class_weight, verbose=0)
            prob = m.predict(X_tr[va_idx], verbose=0).ravel()
            fold_auc.append(roc_auc_score(y_tr[va_idx], prob))
        rows.append({**params, "mean_cv_auc": np.mean(fold_auc), "std_cv_auc": np.std(fold_auc)})
        print(f"  CV {params} -> AUROC {np.mean(fold_auc):.4f} (+/- {np.std(fold_auc):.4f})")
    cv_df = pd.DataFrame(rows).sort_values("mean_cv_auc", ascending=False)
    cv_df.to_csv(f"{OUT_DIR}/cnn_cv_results.csv", index=False)
    best = cv_df.iloc[0]
    return {"filters": int(best["filters"]), "dropout": float(best["dropout"]),
            "lr": float(best["lr"])}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    # ---------- Task 1 ----------
    print("=== Task 1: Data acquisition and exploration ===")
    paths, labels = collect_paths(DATA_DIR)
    report_balance(labels)
    show_samples(paths, labels, n=6)

    # ---------- Task 2 ----------
    print("\n=== Task 2: Preprocessing and feature extraction ===")
    print(f"Resizing {len(paths)} images to {IMG_SIZE}x{IMG_SIZE} grayscale ...")
    images = np.array([load_image(p) for p in paths])

    # 70 / 15 / 15 stratified split (indices, so images and features stay aligned)
    idx = np.arange(len(labels))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=labels[tmp_idx],
                                      random_state=SEED)
    print(f"Split -> train: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)}")

    print("Extracting HOG + LBP + GLCM features ...")
    feats = extract_all(images)
    print("Feature vector length per image:", feats.shape[1])

    scaler = StandardScaler().fit(feats[tr_idx])            # fit on train only
    F_tr, F_va, F_te = (scaler.transform(feats[i]) for i in (tr_idx, va_idx, te_idx))
    y_tr, y_va, y_te = labels[tr_idx], labels[va_idx], labels[te_idx]

    # CNN inputs (pixels scaled to 0-1, channel axis added)
    to_cnn = lambda a: (a.astype("float32") / 255.0)[..., None]
    X_tr, X_va, X_te = to_cnn(images[tr_idx]), to_cnn(images[va_idx]), to_cnn(images[te_idx])

    # ---------- Task 3 ----------
    print("\n=== Task 3: Model training and evaluation ===")
    results_val, results_test, probs_test = [], [], {}

    # --- 3a. CNN baseline with cross-validated hyperparameters
    n_neg, n_pos = np.bincount(y_tr)
    class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}
    print("\n[CNN] Cross-validating hyperparameters ...")
    best_params = cnn_cross_validation(X_tr, y_tr, class_weight)
    print("[CNN] Best params:", best_params)

    keras.backend.clear_session()
    cnn = build_cnn(**best_params)
    hist = cnn.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=CNN_FINAL_EPOCHS, batch_size=32,
        class_weight=class_weight, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                                 restore_best_weights=True)])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("CNN loss"); ax[0].legend()
    ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
    ax[1].set_title("CNN accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_cnn_training_curves.png", dpi=120); plt.show()

    p_va = cnn.predict(X_va, verbose=0).ravel()
    p_te = cnn.predict(X_te, verbose=0).ravel()
    results_val.append(compute_metrics("CNN", y_va, p_va))
    results_test.append(compute_metrics("CNN", y_te, p_te))
    probs_test["CNN"] = p_te

    # --- 3b. Classical classifiers on HOG+LBP+GLCM features (5-fold CV grid search)
    candidates = {
        "Logistic Regression (features)": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"C": [0.01, 0.1, 1]}),
        "SVM-RBF (features)": (
            SVC(probability=True, class_weight="balanced", random_state=SEED),
            {"C": [1, 10], "gamma": ["scale", 0.001]}),
        "Random Forest (features)": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 20]}),
        "k-NN (features)": (
            KNeighborsClassifier(n_jobs=-1),
            {"n_neighbors": [3, 7, 15]}),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, (est, grid) in candidates.items():
        print(f"\n[{name}] grid search ...")
        gs = GridSearchCV(est, grid, cv=skf, scoring="roc_auc", n_jobs=-1).fit(F_tr, y_tr)
        print(f"  best params: {gs.best_params_}  (CV AUROC {gs.best_score_:.4f})")
        results_val.append(compute_metrics(name, y_va, gs.predict_proba(F_va)[:, 1]))
        pt = gs.predict_proba(F_te)[:, 1]
        results_test.append(compute_metrics(name, y_te, pt))
        probs_test[name] = pt

    # --- 3c. Comparison
    val_df = pd.DataFrame(results_val).set_index("Model").round(4)
    test_df = pd.DataFrame(results_test).set_index("Model").round(4)
    val_df.to_csv(f"{OUT_DIR}/results_validation.csv")
    test_df.to_csv(f"{OUT_DIR}/results_test.csv")
    print("\n=== Validation set results ===\n", val_df)
    print("\n=== Test set results ===\n", test_df)

    # Best model is chosen on the VALIDATION set (test set stays untouched for selection)
    best_name = val_df["F1"].idxmax()
    print(f"\nBest performer (highest validation F1): {best_name}")
    print(f"Its test metrics:\n{test_df.loc[best_name]}")

    # ROC curves on test
    plt.figure(figsize=(7, 6))
    for name, pr in probs_test.items():
        fpr, tpr, _ = roc_curve(y_te, pr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, pr):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_roc_curves.png", dpi=120); plt.show()

    # ---------- Confusion matrix and misclassification analysis ----------
    cm = plot_confusion(y_te, probs_test[best_name], f"Confusion matrix - {best_name} (test)",
                        "05_confusion_matrix_best.png")
    plot_confusion(y_te, probs_test["CNN"], "Confusion matrix - CNN (test)",
                   "06_confusion_matrix_cnn.png")

    tn, fp, fn, tp = cm.ravel()
    print("\n=== Misclassification analysis (best model) ===")
    print(f"True Negatives : {tn}  (normal correctly identified)")
    print(f"False Positives: {fp}  (normal predicted as pneumonia -> unnecessary follow-up)")
    print(f"False Negatives: {fn}  (pneumonia missed -> the most dangerous error)")
    print(f"True Positives : {tp}  (pneumonia correctly identified)")
    print("Likely causes: (1) class imbalance favours the PNEUMONIA class, so borderline normal "
          "scans get flagged (false positives); (2) mild or early pneumonia shows faint opacities "
          "that resemble normal lung texture (false negatives); (3) variation in exposure, patient "
          "positioning and scanner quality; (4) 128x128 resizing loses fine detail; "
          "(5) the small open-source label set may contain noisy labels.")

    # Show misclassified test images for the best model
    y_pred = (probs_test[best_name] >= 0.5).astype(int)
    wrong = np.where(y_pred != y_te)[0][:8]
    if len(wrong):
        fig, axes = plt.subplots(2, 4, figsize=(11, 6))
        for ax, w in zip(axes.ravel(), wrong):
            ax.imshow(images[te_idx[w]], cmap="gray"); ax.axis("off")
            ax.set_title(f"True: {CLASS_NAMES[y_te[w]]}\nPred: {CLASS_NAMES[y_pred[w]]} "
                         f"({probs_test[best_name][w]:.2f})", fontsize=9)
        for ax in axes.ravel()[len(wrong):]:
            ax.axis("off")
        plt.suptitle(f"Misclassified test images - {best_name}")
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/07_misclassified.png", dpi=120); plt.show()

    print(f"\nAll plots and tables saved in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
