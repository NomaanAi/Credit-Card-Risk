import nbformat as nbf
import os

def create_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

def main():
    os.makedirs('notebooks', exist_ok=True)
    
    # --- 01 EDA ---
    cells_eda = [
        nbf.v4.new_markdown_cell("# 01. Exploratory Data Analysis\nIn this notebook, we load the raw data and perform initial data exploration."),
        nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nimport os\n\n# Add parent directory to path to import src\nsys.path.append('..')\nfrom src.data.load_data import load_raw_data"),
        nbf.v4.new_code_cell("df = load_raw_data('../data/raw/credit_default_uci.xls')\ndf.head()"),
        nbf.v4.new_code_cell("df.info()"),
        nbf.v4.new_code_cell("df.describe()"),
        nbf.v4.new_code_cell("# Check target distribution\nsns.countplot(x='target', data=df)\nplt.title('Target Distribution')\nplt.show()"),
        nbf.v4.new_code_cell("# Correlation Matrix\nplt.figure(figsize=(12, 10))\nsns.heatmap(df.corr(), cmap='coolwarm')\nplt.title('Correlation Matrix')\nplt.show()")
    ]
    create_notebook('notebooks/01_eda.ipynb', cells_eda)

    # --- 02 Preprocessing ---
    cells_prep = [
        nbf.v4.new_markdown_cell("# 02. Preprocessing\nCleaning and checking data splits."),
        nbf.v4.new_code_cell("import sys\nsys.path.append('..')\nfrom src.data.load_data import load_raw_data\nfrom src.data.split_data import split_train_test\nfrom src.features.preprocessing import preprocess_data"),
        nbf.v4.new_code_cell("df = load_raw_data('../data/raw/credit_default_uci.xls')\nX_train, X_test, y_train, y_test = split_train_test(df)\nprint(f'Train shape: {X_train.shape}')\nprint(f'Test shape: {X_test.shape}')"),
        nbf.v4.new_code_cell("# Verify Preprocessing pipeline\nX_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)\nprint('Processed Train shape:', X_train_proc.shape)")
    ]
    create_notebook('notebooks/02_preprocessing.ipynb', cells_prep)

    # --- 03 Modeling Baselines ---
    cells_base = [
        nbf.v4.new_markdown_cell("# 03. Baseline Modeling\nLogistic Regression."),
        nbf.v4.new_code_cell("import sys\nsys.path.append('..')\nfrom src.data.load_data import load_raw_data\nfrom src.data.split_data import split_train_test\nfrom src.features.preprocessing import preprocess_data\nfrom src.models.train_logistic import train_logistic_regression\nfrom src.evaluation.metrics import calculate_metrics"),
        nbf.v4.new_code_cell("# Load and Prep\ndf = load_raw_data('../data/raw/credit_default_uci.xls')\nX_train, X_test, y_train, y_test = split_train_test(df)\nX_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)"),
        nbf.v4.new_code_cell("model = train_logistic_regression(X_train_proc, y_train)\ny_pred = model.predict(X_test_proc)\ny_prob = model.predict_proba(X_test_proc)[:, 1]"),
        nbf.v4.new_code_cell("metrics = calculate_metrics(y_test, y_pred, y_prob)\nmetrics")
    ]
    create_notebook('notebooks/03_modeling_baselines.ipynb', cells_base)
    
    # --- 04 Modeling Advanced ---
    cells_adv = [
        nbf.v4.new_markdown_cell("# 04. Advanced Modeling\nRandom Forest and Gradient Boosting."),
        nbf.v4.new_code_cell("import sys\nsys.path.append('..')\nfrom src.data.load_data import load_raw_data\nfrom src.data.split_data import split_train_test\nfrom src.features.preprocessing import preprocess_data\nfrom src.models.train_random_forest import train_random_forest\nfrom src.models.train_gradient_boosting import train_gradient_boosting\nfrom src.evaluation.metrics import calculate_metrics"),
        nbf.v4.new_code_cell("# Load and Prep\ndf = load_raw_data('../data/raw/credit_default_uci.xls')\nX_train, X_test, y_train, y_test = split_train_test(df)\nX_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)"),
        nbf.v4.new_code_cell("# Random Forest\nrf = train_random_forest(X_train_proc, y_train, n_estimators=50)\nmetrics_rf = calculate_metrics(y_test, rf.predict(X_test_proc), rf.predict_proba(X_test_proc)[:, 1])\nprint('RF:', metrics_rf)"),
        nbf.v4.new_code_cell("# Gradient Boosting\ngb = train_gradient_boosting(X_train_proc, y_train, n_estimators=50)\nmetrics_gb = calculate_metrics(y_test, gb.predict(X_test_proc), gb.predict_proba(X_test_proc)[:, 1])\nprint('GB:', metrics_gb)")
    ]
    create_notebook('notebooks/04_modeling_advanced.ipynb', cells_adv)

    # --- 05 Evaluation ---
    cells_eval = [
        nbf.v4.new_markdown_cell("# 05. Evaluation & Calibration\nAnalyzing model performance and calibration."),
        nbf.v4.new_code_cell("import sys\nimport matplotlib.pyplot as plt\nsys.path.append('..')\nfrom src.data.load_data import load_raw_data\nfrom src.data.split_data import split_train_test\nfrom src.features.preprocessing import preprocess_data\nfrom src.models.train_logistic import train_logistic_regression\nfrom src.evaluation.calibration import plot_calibration_curve_func"),
        nbf.v4.new_code_cell("df = load_raw_data('../data/raw/credit_default_uci.xls')\nX_train, X_test, y_train, y_test = split_train_test(df)\nX_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)"),
        nbf.v4.new_code_cell("lr = train_logistic_regression(X_train_proc, y_train)\ny_prob = lr.predict_proba(X_test_proc)[:, 1]"),
        nbf.v4.new_code_cell("plot_calibration_curve_func(y_test, y_prob, 'Logistic Regression')\nplt.show()")
    ]
    create_notebook('notebooks/05_evaluation_calibration.ipynb', cells_eval)

    # --- 06 Interpretability ---
    cells_interp = [
        nbf.v4.new_markdown_cell("# 06. Interpretability\nFeature Importance and SHAP."),
        nbf.v4.new_code_cell("import sys\nimport matplotlib.pyplot as plt\nsys.path.append('..')\nfrom src.data.load_data import load_raw_data\nfrom src.data.split_data import split_train_test\nfrom src.features.preprocessing import preprocess_data\nfrom src.models.train_random_forest import train_random_forest\nfrom src.interpretability.feature_importance import plot_feature_importance"),
        nbf.v4.new_code_cell("df = load_raw_data('../data/raw/credit_default_uci.xls')\nX_train, X_test, y_train, y_test = split_train_test(df)\nX_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)"),
        nbf.v4.new_code_cell("rf = train_random_forest(X_train_proc, y_train, n_estimators=50)"),
        nbf.v4.new_code_cell("feat_names = [f'feat_{i}' for i in range(X_train_proc.shape[1])]\nplot_feature_importance(rf, feat_names)\nplt.show()")
    ]
    create_notebook('notebooks/06_interpretability_shap.ipynb', cells_interp)

if __name__ == "__main__":
    main()
