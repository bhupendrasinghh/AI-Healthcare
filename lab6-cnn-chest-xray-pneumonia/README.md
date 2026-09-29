# Lab 6 — CNN + Handcrafted Features on Chest X-Ray Images (Normal vs Pneumonia)

**Course:** CSET343 - AI in Healthcare  
**Topic:** Deep Learning · CNN · Computer Vision · Medical Imaging  
**Author:** Bhupendra Singh

---

## 📋 Overview

This lab trains a **Convolutional Neural Network (CNN)** combined with handcrafted image features (**HOG**, **LBP**, **GLCM**) to classify chest X-ray images as **Normal** or **Pneumonia**. It explores both traditional feature engineering and deep learning approaches for medical image analysis.

---

## 🗂️ Folder Structure

```
lab6-cnn-chest-xray-pneumonia/
├── lab6_chest_xray_cnn.py   # Main script
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## 📥 Dataset Setup

Download the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:  
🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract it so the structure looks like:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Then set `DATA_DIR` in the script to point to the `chest_xray/` folder.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the script
python lab6_chest_xray_cnn.py
```

---

## 🧪 Key Concepts Covered

| Concept | Detail |
|---|---|
| Dataset | Chest X-Ray Images (Kaggle - Paul Mooney) |
| Task | Binary Classification (Normal / Pneumonia) |
| Features | HOG, LBP, GLCM (handcrafted) + CNN (deep) |
| Algorithms | CNN (TensorFlow/Keras), Logistic Regression, SVM |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `tensorflow` | CNN model building & training |
| `scikit-learn` | Classical ML models & metrics |
| `scikit-image` | HOG, LBP, GLCM feature extraction |
| `Pillow` | Image loading & preprocessing |
| `matplotlib` | Plotting |
| `pandas` / `numpy` | Data manipulation |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
