from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from numpy import trapz

def compute_auc_pauc(y_true, y_scores, fpr_threshold=0.1):
    auc = roc_auc_score(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = fpr <= fpr_threshold
    pauc = trapz(tpr[mask], fpr[mask]) / fpr_threshold

    return auc, pauc