import pandas as pd
import os
import joblib
from src.data.load_data import load_raw_data
from src.data.split_data import split_train_test
from src.features.preprocessing import preprocess_data
from src.models.train_logistic import train_logistic_regression
from src.models.train_random_forest import train_random_forest
from src.models.train_gradient_boosting import train_gradient_boosting
from src.evaluation.metrics import calculate_metrics
from src.evaluation.calibration import plot_calibration_curve_func
from src.interpretability.feature_importance import plot_feature_importance
from src.interpretability.shap_analysis import run_shap_analysis

def main():
    print("1. Loading Data...")
    raw_filepath = "data/raw/credit_default_uci.xls"
    df = load_raw_data(raw_filepath)
    print(f"   Shape: {df.shape}")

    print("2. Splitting Data...")
    X_train, X_test, y_train, y_test = split_train_test(df)
    print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("3. Preprocessing...")
    X_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)
    
    # Get feature names from preprocessor
    # This is a bit tricky with transformers, usually we can extract them
    # For now, we will try to get them or just use indices
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feat_{i}" for i in range(X_train_proc.shape[1])]

    results = {}
    
    # --- Logistic Regression ---
    print("\n4. Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_proc, y_train)
    y_pred_lr = lr_model.predict(X_test_proc)
    y_prob_lr = lr_model.predict_proba(X_test_proc)[:, 1]
    results['Logistic Regression'] = calculate_metrics(y_test, y_pred_lr, y_prob_lr)
    
    # --- Random Forest ---
    print("5. Training Random Forest...")
    rf_model = train_random_forest(X_train_proc, y_train, n_estimators=50) # Reduced for speed
    y_pred_rf = rf_model.predict(X_test_proc)
    y_prob_rf = rf_model.predict_proba(X_test_proc)[:, 1]
    results['Random Forest'] = calculate_metrics(y_test, y_pred_rf, y_prob_rf)

    # --- Gradient Boosting ---
    print("6. Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train_proc, y_train, n_estimators=50)
    y_pred_gb = gb_model.predict(X_test_proc)
    y_prob_gb = gb_model.predict_proba(X_test_proc)[:, 1]
    results['Gradient Boosting'] = calculate_metrics(y_test, y_pred_gb, y_prob_gb)

    print("\n--- Results ---")
    results_df = pd.DataFrame(results).T
    print(results_df)

    # --- Plotting ---
    print("\n7. Generating Plots...")
    os.makedirs("reports/figures", exist_ok=True)
    
    # Calibration
    plot_calibration_curve_func(y_test, y_prob_lr, "Logistic Regression", "reports/figures/calibration_lr.png")
    plot_calibration_curve_func(y_test, y_prob_rf, "Random Forest", "reports/figures/calibration_rf.png")
    
    # Feature Importance (RF)
    plot_feature_importance(rf_model, feature_names, "reports/figures/feature_importance_rf.png")
    
    # SHAP (GB) - On small sample
    print("   Running SHAP (this might take a minute)...")
    # SHAP for trees is fast, but let's be safe with sample
    X_sample = X_test_proc[:100]
    try:
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()
        # run_shap_analysis(gb_model, X_sample, "reports/figures/shap_summary.png")
        print("   Skipping SHAP analysis due to library incompatibility (known issue with recent sklearn).")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    print("\nDone! Check reports/figures/ for outputs.")

if __name__ == "__main__":
    main()
