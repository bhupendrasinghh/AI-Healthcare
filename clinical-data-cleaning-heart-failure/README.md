# Clinical Data Cleaning & Enrichment — Heart Failure Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhupendrasinghh/Clinical-Data-Cleaning-Enrichment-Heart-Failure-Dataset/blob/main/clinical_data_cleaning_heart_failure.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An end-to-end clinical data engineering and data quality pipeline designed for processing tabular healthcare records. Demonstrates data profiling, missingness visualization, skew-aware imputation, IQR outlier remediation, logical inconsistency validation, feature engineering of clinical risk ratios, MinMax normalization, and comprehensive before-vs-after statistical validation.

---

## 🌟 Key Features

- **MCAR Missingness Simulation & Visualization**: Injects controlled missing completely at random (MCAR) missingness into key clinical attributes (`serum_creatinine`, `ejection_fraction`, `smoking`) and visualizes missing patterns using Seaborn heatmaps.
- **Skew-Aware Imputation**: Applies median imputation for continuous skewed clinical parameters (e.g. serum creatinine, platelets) and mode imputation for categorical attributes.
- **IQR Outlier Remediation**: Uses the Interquartile Range (IQR) rule to detect and prune extreme clinical outliers without distorting underlying physiological distributions.
- **Logical Inconsistency & Range Validation**: Enforces clinical domain boundary checks (e.g. non-negative age, physiological bounds for sodium and ejection fraction).
- **Clinical Feature Engineering**: Derives risk indicators including Creatinine-Sodium ratio (`serum_creatinine / serum_sodium`) and Age-Risk stratifications.
- **Categorical Encoding & MinMax Normalization**: One-hot encodes categorical attributes and scales numeric clinical metrics to a bounded $[0, 1]$ interval.
- **Before-vs-After Validation**: Provides comparative statistical summaries and distribution plots (pre vs. post cleaning) to verify data integrity.

---

## 📊 Dataset Details

- **Dataset**: [UCI Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) (*Chicco & Jurman, 2020*)
- **Attributes**: 299 patient records with 13 clinical features (age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, time, DEATH_EVENT).

---

## 🛠 Tech Stack

- **Data Engineering & Analysis**: `pandas`, `numpy`
- **Machine Learning & Preprocessing**: `scikit-learn` (`MinMaxScaler`)
- **Data Visualization**: `matplotlib`, `seaborn`

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.10+ is installed.

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bhupendrasinghh/Clinical-Data-Cleaning-Enrichment-Heart-Failure-Dataset.git
   cd Clinical-Data-Cleaning-Enrichment-Heart-Failure-Dataset
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Jupyter Notebook**:
   ```bash
   jupyter notebook clinical_data_cleaning_heart_failure.ipynb
   ```

---

## 📁 Repository Structure

```
Clinical-Data-Cleaning-Enrichment-Heart-Failure-Dataset/
├── clinical_data_cleaning_heart_failure.ipynb  # Main data cleaning pipeline notebook
├── README.md                                     # Project documentation
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
└── LICENSE                                      # MIT License
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
