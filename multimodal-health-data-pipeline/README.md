# Multimodal Healthcare Data Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhupendrasinghh/Multimodal-Healthcare-Data-Pipeline/blob/main/multimodal_health_pipeline.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A modular end-to-end healthcare data engineering and machine learning pipeline for ingesting, preprocessing, and analyzing heterogeneous medical data across four primary modalities: **structured clinical labs (tabular)**, **physiological signals (ECG)**, **clinical notes (unstructured text)**, and **medical imaging (chest X-rays)**.

---

## 🌟 Key Features

- **Per-Modality Modular Loaders**: Individual ingestion modules designed with format-specific parsing, schema validation, and data cleaning for each healthcare data type.
- **Graceful Fallback to Synthetic Data**: Embedded fallback generators allow the entire ingestion and modeling pipeline to execute reliably even in offline or network-restricted environments.
- **Tabular Heart Disease Baseline Model**: Ingests the 14-feature UCI Cleveland Heart Disease dataset, handles missing entries, normalizes continuous features, and trains a Logistic Regression risk classifier with ROC-AUC evaluation.
- **ECG Signal & HRV Feature Extraction**: Parses MIT-BIH Arrhythmia database WFDB format triplets (`.dat`/`.hea`/`.atr`), aligns R-peak annotations, and calculates Heart Rate Variability (HRV / RR-intervals).
- **Clinical Text Processing**: Natural Language Processing (NLP) pipeline for clinical notes including tokenization, stopword filtering, and term frequency extraction.
- **Medical Image Pipeline**: Preprocesses medical images (chest X-rays), handles resolution normalization (resizing to standard 224x224), and extracts pixel distribution statistics.

---

## 📊 Datasets Used

1. **UCI Cleveland Heart Disease Dataset (Tabular)**
   - *Source*: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
   - *Description*: 303 patient instances with 14 clinical features (age, sex, chest pain type, resting blood pressure, serum cholesterol, etc.).
2. **MIT-BIH Arrhythmia Database (Signal)**
   - *Source*: [PhysioNet MIT-BIH Database](https://physionet.org/content/mitdb/1.0.0/) via `wfdb`
   - *Description*: Two-channel ambulatory ECG recordings with cardiac beat annotations.
3. **Synthetic Clinical Notes & Chest X-Rays**
   - *Description*: Privacy-compliant synthetic data for demonstrating NLP tokenization and image preprocessing workflows.

---

## 🛠 Tech Stack

- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Signal Processing**: `wfdb`, `scipy`
- **Natural Language Processing**: `nltk`
- **Image Processing**: `Pillow` (PIL)
- **Data Visualization**: `matplotlib`, `seaborn`

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.10+ is installed on your system.

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bhupendrasinghh/Multimodal-Healthcare-Data-Pipeline.git
   cd multimodal-health-data-pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Jupyter Notebook**:
   ```bash
   jupyter notebook multimodal_health_pipeline.ipynb
   ```

---

## 📁 Repository Structure

```
multimodal-health-data-pipeline/
├── multimodal_health_pipeline.ipynb  # Core Jupyter notebook containing the full pipeline
├── README.md                          # Project documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── LICENSE                           # MIT License
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
