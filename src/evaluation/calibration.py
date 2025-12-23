from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curve_func(y_true, y_prob, model_name="Model", save_path=None):
    """
    Plot calibration curve.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    # plt.show() # Avoid showing if running in script mode
    return plt
