# Credit Risk Machine Learning Project

This project focuses on predicting credit card default using machine learning techniques. It utilizes the **UCI Credit Card Default** dataset to build, evaluate, and interpret predictive models. The goal is to identify high-risk customers and understand the key drivers of credit default.

## 📌 Tables of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## 🚀 Project Overview
Credit default prediction is a critical task for financial institutions. By leveraging historical data on customer behavior and demographics, we can build models to estimate the probability of default. This project implements a complete data science pipeline:
1.  **Exploratory Data Analysis (EDA)** to understand data distribution.
2.  **Preprocessing** for cleaning and feature engineering.
3.  **Model Training** using baselines and advanced ensemble methods.
4.  **Evaluation** using appropriate metrics (ROC-AUC, Precision-Recall) and calibration.
5.  **Interpretability** using SHAP values to explain model decisions.

## 📊 Dataset
The dataset used is the [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from the UCI Machine Learning Repository.

- **Instances**: 30,000
- **Attributes**: 24 (Demographics, Payment History, Bill Statements, Previous Payments)
- **Target**: `default.payment.next.month` (1 = Default, 0 = No Default)

*Note: The raw data file is located at `data/raw/credit_default_uci.xls`.*

## 📂 Project Structure
```
credit-risk-ml/
├── data/
│   ├── raw/                 # Original immutable data
│   └── processed/           # Processed data structures (train/test splits)
├── notebooks/               # Jupyter notebooks for interactive development
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb          # Cleaning and Feature Engineering
│   ├── 03_modeling_baselines.ipynb     # Logistic Regression & Simple Models
│   ├── 04_modeling_advanced.ipynb      # Random Forest & Gradient Boosting
│   ├── 05_evaluation_calibration.ipynb # Performance Metrics & Calibration Curves
│   └── 06_interpretability_shap.ipynb  # SHAP Analysis & Feature Importance
├── src/                     # Source code for reproduction
│   ├── data/                # Scripts to load and split data
│   ├── features/            # Preprocessing pipelines
│   ├── models/              # Training scripts
│   ├── evaluation/          # Metric calculations & plotting
│   └── interpretability/    # Explainability tools
├── reports/                 # Generated analysis
│   └── figures/             # PNG/SVG plots
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Running Notebooks
The notebooks are designed to be run in sequential order:
1.  Start Jupyter:
    ```bash
    jupyter notebook
    ```
2.  Open `notebooks/01_eda.ipynb` to explore the data.
3.  Proceed through 02-06 to replicate the full pipeline.

### Source Code
You can also import functions from the `src` package for use in your own scripts:
```python
from src.data.load_data import load_raw_data
from src.features.preprocessing import preprocess_data

df = load_raw_data("data/raw/credit_default_uci.xls")
df_clean = preprocess_data(df)
```

## 🔬 Methodology

### Models
- **Logistic Regression**: A baseline linear model for interpretability.
- **Random Forest**: An ensemble method to capture non-linear relationships.
- **Gradient Boosting (XGBoost/LightGBM)**: High-performance boosting algorithms for state-of-the-art results.

### Evaluation
We use metrics that are robust to class imbalance:
- **ROC-AUC Score**
- **Precision-Recall Curve**
- **F1-Score**
- **Brier Score** (for probability calibration)

### Interpretability
- **Global Importance**: Permutation importance and feature split counts.
- **Local Importance**: SHAP (SHapley Additive exPlanations) values to explain individual predictions.

## 📈 Results
*Results section will be updated after running the modeling notebooks.*

## 📝 License
This project is open-source.
