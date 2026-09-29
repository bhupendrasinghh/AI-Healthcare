# Multimodal Medical Data Acquisition, Cleaning & Preprocessing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhupendrasinghh/Multimodal-Medical-Data-Acquisition-Cleaning-Preprocessing/blob/main/multimodal_medical_data_acquisition_cleaning_preprocessing.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An end-to-end data engineering and data quality pipeline designed specifically for healthcare AI systems across four primary medical data modalities: **structured clinical labs (tabular)**, **unstructured clinical notes (text)**, **medical imaging (chest X-rays / DICOM)**, and **physiological signals (ECG)**.

---

## 🌟 Key Features

### Module 1 — Structured Clinical Tabular Data
- **Implausible Zero-Value Detection**: Identifies medically impossible zero readings (e.g. 0 Blood Pressure, 0 Glucose, 0 BMI) as implicit missingness.
- **Class-Stratified Median Imputation**: Imputes missing entries using class-specific medians (diabetic vs. non-diabetic sub-populations) to preserve physiological signals.
- **Outlier Remediation & Feature Scaling**: Applies z-score and IQR filtering alongside StandardScaler & MinMaxScaler.
- **Inferential Statistics & Feature Selection**: Conducts Chi-Square tests of independence, independent two-sample t-tests, ANOVA F-rankings, and Random Forest feature importance scoring.

### Module 2 — Unstructured Clinical Text Data
- **PHI De-identification**: Redacts Protected Health Information (patient names, medical record numbers, dates, phone numbers) using automated regex pattern matching.
- **NLP Text Normalization**: Cleans clinical text via lowercasing, punctuation stripping, tokenization, and stopword removal using `nltk`.
- **TF-IDF Feature Extraction**: Converts unstructured clinical narratives into numerical TF-IDF term-frequency matrices for downstream modeling.

### Module 3 — Medical Imaging Data
- **DICOM / Image Anonymization**: Strips patient metadata embedded in DICOM headers.
- **Contrast Enhancement**: Applies Histogram Equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE) to reveal subtle anatomical structures.
- **Noise Reduction & Normalization**: Filters high-frequency spatial noise with Gaussian blurring, normalizes pixel intensities to $[0, 1]$, and resizes images to standard $224 \times 224$ dimensions.

### Module 4 — Physiological Signal Data
- **Ambulatory ECG Signal Ingestion**: Parses PhysioNet `wfdb` format ECG signals (`.dat`/`.hea`/`.atr`).
- **Butterworth Bandpass Filtering**: Applies a 4th-order Butterworth bandpass filter ($0.5-40\text{ Hz}$) to eliminate baseline wander, muscle artifacts, and powerline noise.
- **R-Peak Alignment & HRV Feature Extraction**: Detects R-peaks, segments cardiac cycles, and computes Heart Rate Variability (HRV) metrics including mean RR-intervals, RMSSD, and LF/HF spectral ratios.

---

## 📊 Datasets Used

1. **Pima Indians Diabetes Dataset (Tabular)**
   - *Source*: [Pima Indians Diabetes Data](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
   - *Description*: 768 patient instances with 8 diagnostic features and 1 target outcome class.
2. **Synthetic Clinical Notes (Text)**
   - *Description*: Privacy-compliant clinical narratives generated for demonstrating PHI de-identification and TF-IDF extraction.
3. **Chest X-Ray Imaging (Image)**
   - *Description*: Public medical X-ray dataset utilized for demonstrating spatial filtering, CLAHE contrast tuning, and pixel scaling.
4. **MIT-BIH Arrhythmia Database (Signal)**
   - *Source*: [PhysioNet MIT-BIH Database](https://physionet.org/content/mitdb/1.0.0/) via `wfdb`
   - *Description*: Two-channel ambulatory ECG signals with expert cardiac beat annotations.

---

## 🛠 Tech Stack

- **Data Engineering**: `pandas`, `numpy`
- **Machine Learning & Inferential Stats**: `scikit-learn`, `scipy`
- **Natural Language Processing**: `nltk`
- **Medical Image Processing**: `Pillow`, `opencv-python`
- **Physiological Signal Processing**: `wfdb`
- **Data Visualization**: `matplotlib`, `seaborn`

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.10+ is installed.

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bhupendrasinghh/Multimodal-Medical-Data-Acquisition-Cleaning-Preprocessing.git
   cd Multimodal-Medical-Data-Acquisition-Cleaning-Preprocessing
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Jupyter Notebook**:
   ```bash
   jupyter notebook multimodal_medical_data_acquisition_cleaning_preprocessing.ipynb
   ```

---

## 📁 Repository Structure

```
Multimodal-Medical-Data-Acquisition-Cleaning-Preprocessing/
├── multimodal_medical_data_acquisition_cleaning_preprocessing.ipynb  # Core pipeline notebook
├── README.md                                                          # Project documentation
├── requirements.txt                                                  # Python dependencies
├── .gitignore                                                        # Git ignore rules
└── LICENSE                                                           # MIT License
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
