from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities. (for class 1)
        
    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob)
    }
    return metrics
