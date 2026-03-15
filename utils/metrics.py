import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

def calculate_classification_metrics(y_true, y_pred):
    """
    Precision, Recall, F1 skorlarını hesaplayalım
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"precision":precision, "recall":recall, "f1_score": f1}

def get_classification_report(y_true, y_pred):
    """ Classification report metni"""
    return classification_report(y_true, y_pred, zero_division=0)

def confusion_matrix_val(y_true,y_pred):
    return confusion_matrix(y_true, y_pred)

def calculate_roc_auc(y_true, y_prob):
    """ Roc-Auc hesaplanır"""
    return roc_auc_score(y_true, y_prob)

def find_best_threshold_by_f1(y_true, y_prob, thresholds=None):
    """ Farklı threshold değerlerini dener ve en iyi F1 skorunu veren thresholdu bulmaya çalışır"""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    results = []
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1, results
