"""Metrics and raw predictions. Labels and confusion-matrix order are fixed."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data_loader import LABELS


def evaluate(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS,
                                    output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "per_class": {label: report[label] for label in LABELS},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            "label_order": LABELS}
