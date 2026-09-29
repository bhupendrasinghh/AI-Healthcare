# Lab 7 — Grid Search on a Deep Learning Model (Pima Indians Diabetes)

**Course:** CSET343 - AI in Healthcare  
**Topic:** Deep Learning · Hyperparameter Tuning · Grid Search  
**Author:** Bhupendra Singh

---

## 📋 Overview

This lab performs **Grid Search hyperparameter tuning** on a **Keras deep learning model** using the **Pima Indians Diabetes Dataset**. It wraps the Keras model with `scikeras` to integrate seamlessly with scikit-learn's `GridSearchCV`, exploring different architectures, optimisers, and regularisation strategies.

---

## 🗂️ Folder Structure

```
lab7-grid-search-deep-learning-diabetes/
├── lab7_grid_search_pima.py   # Main script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python lab7_grid_search_pima.py
```

The dataset is **automatically downloaded** from GitHub (Jason Brownlee's repository).

---

## 🧪 Key Concepts Covered

| Concept | Detail |
|---|---|
| Dataset | Pima Indians Diabetes (auto-downloaded) |
| Task | Binary Classification (Diabetic / Non-Diabetic) |
| Model | Keras Neural Network |
| Tuning | GridSearchCV via `scikeras` |
| Search Space | Layers, neurons, optimiser, batch size, epochs |
| Metrics | Accuracy, Classification Report |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `tensorflow` | Deep learning model (Keras) |
| `scikeras` | Keras ↔ scikit-learn wrapper |
| `scikit-learn` | GridSearchCV, metrics |
| `pandas` / `numpy` | Data loading & manipulation |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
