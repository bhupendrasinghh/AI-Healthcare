# Lab 5 — Multiclass Classification on the Dermatology Dataset (UCI)

**Course:** CSET343 - AI in Healthcare  
**Topic:** Multiclass Classification · Multiple ML Algorithms  
**Author:** Bhupendra Singh

---

## 📋 Overview

This lab performs **multiclass skin disease classification** using the **UCI Dermatology Dataset**. Multiple classifiers are compared — Logistic Regression, K-Nearest Neighbours, Random Forest, SVM, and Decision Tree — to identify six dermatological conditions.

---

## 🗂️ Folder Structure

```
lab5-multiclass-classification-dermatology/
├── lab5_multiclass_dermatology.py   # Main script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python lab5_multiclass_dermatology.py
```

Plots will be saved to `./lab5_outputs/`.

---

## 🧪 Key Concepts Covered

| Concept | Detail |
|---|---|
| Dataset | UCI Dermatology Dataset (auto-downloaded) |
| Task | Multiclass Classification (6 skin conditions) |
| Algorithms | Logistic Regression, KNN, Random Forest, SVM, Decision Tree |
| Metrics | Accuracy, Precision, Recall, F1 (macro), ROC-AUC |
| Visualisations | Confusion Matrices, ROC Curves, Model Comparison |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models & metrics |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisation |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
