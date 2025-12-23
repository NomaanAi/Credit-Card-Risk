import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 20
        top_k = 20
        indices = indices[:top_k]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        return indices
    else:
        print("Model does not have feature_importances_ attribute.")
        return None
