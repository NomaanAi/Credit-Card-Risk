from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

def analyze_thresholds(y_true, y_prob, save_path=None):
    """
    Analyze optimal thresholds using Precision-Recall Curve.
    
    Returns:
        best_threshold, best_f1
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # epsilon to avoid div by zero
    
    # Find the index of the best F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='black', label='Best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Best F1={best_f1:.2f} at Thresh={best_threshold:.2f})')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        
    return best_threshold, best_f1
