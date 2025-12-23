import shap
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_sample, save_path=None):
    """
    Run SHAP analysis over a sample of data.
    """
    # Create object that can calculate shap values
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
