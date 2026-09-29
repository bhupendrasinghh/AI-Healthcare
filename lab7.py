"""
CSET343 - AI in Healthcare | Lab Assignment 7
Grid Search on a Deep Learning Model (Pima Indians Diabetes)

Install:  pip install tensorflow scikeras scikit-learn pandas numpy
Run:      python lab7_grid_search_pima.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------
# Task 1: Data acquisition and loading
# ---------------------------------------------------------------
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(URL, names=COLS)   # replace URL with a local path if needed
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df["Outcome"].value_counts())

# ---------------------------------------------------------------
# Task 2: Data preprocessing
# ---------------------------------------------------------------
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
print("\nMissing values:\n", df.isnull().sum())
df[zero_as_missing] = df[zero_as_missing].fillna(df[zero_as_missing].median())

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only (no leakage)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------
# Task 3: Deep learning grid search
# ---------------------------------------------------------------
def build_model(hidden_neurons=12, activation="relu",
                init_mode="glorot_uniform", dropout_rate=0.0, meta=None):
    n_features = meta["n_features_in_"]   # supplied automatically by SciKeras
    return keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(hidden_neurons, activation=activation,
                     kernel_initializer=init_mode),
        layers.Dropout(dropout_rate),
        layers.Dense(8, activation=activation, kernel_initializer=init_mode),
        layers.Dense(1, activation="sigmoid", kernel_initializer=init_mode),
    ])


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def make_clf():
    return KerasClassifier(
        model=build_model,
        loss="binary_crossentropy",
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=50,
        batch_size=32,
        verbose=0,
        random_state=42,
    )


def run_grid(param_grid, **fixed):
    """Run GridSearchCV, print results, return best params."""
    clf = make_clf().set_params(**fixed)
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring="accuracy", n_jobs=1)
    result = grid.fit(X_train, y_train)

    print(f"\nBest: {result.best_score_:.4f} using {result.best_params_}")
    means = result.cv_results_["mean_test_score"]
    stds = result.cv_results_["std_test_score"]
    for m, s, p in zip(means, stds, result.cv_results_["params"]):
        print(f"{m:.4f} ({s:.4f}) with {p}")
    return result.best_params_


def valid_activations(names):
    """Drop activations not available in the installed Keras version."""
    ok = []
    for n in names:
        try:
            keras.activations.get(n)
            ok.append(n)
        except Exception:
            print(f"Skipping unavailable activation: {n}")
    return ok


if __name__ == "__main__":
    # Baseline before tuning
    base = make_clf().fit(X_train, y_train)
    base_acc = accuracy_score(y_test, base.predict(X_test))
    print("\nBaseline test accuracy:", base_acc)

    best = {}

    # 3.3 Batch size and epochs
    print("\n=== Tune batch size and epochs ===")
    best.update(run_grid({"batch_size": [10, 20, 40, 60, 80, 100],
                          "epochs": [10, 50, 100]}))

    # 3.4 Learning rate and momentum (SGD)
    print("\n=== Tune learning rate and momentum ===")
    best.update(run_grid({"optimizer__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
                          "optimizer__momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]},
                         optimizer="SGD", **best))

    # 3.5 Weight initialization
    print("\n=== Tune weight initialization ===")
    best.update(run_grid({"model__init_mode": [
        "uniform", "lecun_uniform", "normal", "zero",
        "glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]},
        optimizer="SGD", **best))

    # 3.6 Activation function
    print("\n=== Tune activation function ===")
    acts = valid_activations(["softmax", "softplus", "softsign", "relu",
                              "tanh", "sigmoid", "hard_sigmoid", "linear"])
    best.update(run_grid({"model__activation": acts}, optimizer="SGD", **best))

    # 3.7 Dropout
    print("\n=== Tune dropout regularization ===")
    best.update(run_grid({"model__dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4,
                                                  0.5, 0.6, 0.7, 0.8, 0.9]},
                         optimizer="SGD", **best))

    # 3.8 Hidden neurons
    print("\n=== Tune number of hidden neurons ===")
    best.update(run_grid({"model__hidden_neurons": [1, 5, 10, 15, 20, 25, 30]},
                         optimizer="SGD", **best))

    print("\nFinal tuned hyperparameters:", best)

    # 3.9 Final model
    final = make_clf().set_params(optimizer="SGD", **best)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)

    print("\nBaseline test accuracy:", base_acc)
    print("Tuned test accuracy   :", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred,
                                target_names=["No Diabetes", "Diabetes"]))
