"""
Train a baseline Logistic Regression (multiclass) on sel_all.csv.

Inputs
------
- /src/sel_all.csv   (features + 'Risk_Level' target)

Outputs (written to /reports)
-----------------------------
- logreg_baseline_classification_report.txt
- logreg_baseline_confusion_matrix.png
- logreg_baseline_roc_ovr.png
- logreg_baseline_pr_ovr.png

Notes
-----
- Uses a single 70/30 hold-out split (test_size=0.30, random_state=42, stratify=y).
- Multinomial Logistic Regression with class_weight='balanced'.
- Prints accuracy, macro/weighted precision/recall/F1, ROC-AUC macro (OvR), and
  Average Precision (macro OvR) to the console.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

# --------------------
# Config
# --------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
CLASS_WEIGHT = "balanced"
MAX_ITER = 2000
SOLVER = "lbfgs"          
MULTI_CLASS = "multinomial"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sel_all.csv"
REPORTS_DIR = BASE_DIR.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Helpers
# --------------------
def load_xy(path: Path):
    df = pd.read_csv(path)
    if "Risk_Level" not in df.columns:
        raise ValueError("Column 'Risk_Level' not found in sel_all.csv")
    X = df.drop(columns=["Risk_Level"])
    y = df["Risk_Level"].astype(int)
    return X, y

def plot_confusion(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc_ovr(y_true, proba, classes, out_path: Path):
    y_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(7, 6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {c}")
    plt.plot([0, 1], [0, 1], "--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_pr_ovr(y_true, proba, classes, out_path: Path):
    y_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(7, 6))
    for i, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        plt.plot(rec, prec, label=f"Class {c}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (One-vs-Rest)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# --------------------
# Main
# --------------------
def main():
    # Load data
    X, y = load_xy(DATA_PATH)

    # Fixed hold-out split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3) Pipeline: scale -> multinomial Logistic Regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),  # features are numeric/OHE, scaling helps LR
        ("clf", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            solver=SOLVER,
            class_weight=CLASS_WEIGHT
        ))
    ])

    pipe.fit(X_train, y_train)

    # Predictions & probabilities
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)   # shape (n_samples, n_classes)
    classes_sorted = np.sort(y.unique())

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    p_ma = precision_score(y_test, y_pred, average="macro",   zero_division=0)
    r_ma = recall_score( y_test, y_pred, average="macro",     zero_division=0)
    f1_ma= f1_score(     y_test, y_pred, average="macro",     zero_division=0)
    p_w  = precision_score(y_test, y_pred, average="weighted",zero_division=0)
    r_w  = recall_score( y_test, y_pred, average="weighted",  zero_division=0)
    f1_w = f1_score(     y_test, y_pred, average="weighted",  zero_division=0)

    auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    # Average Precision
    y_test_bin = pd.get_dummies(y_test).reindex(columns=classes_sorted, fill_value=0)
    ap_macro   = average_precision_score(y_test_bin, y_proba, average="macro")

    # Console summary
    print("=== Logistic Regression Baseline (70/30 hold-out) ===")
    print(f"Samples: train={len(y_train)}  test={len(y_test)}")
    print(f"Accuracy           : {acc:.3f}")
    print(f"Precision (macro)  : {p_ma:.3f}")
    print(f"Recall (macro)     : {r_ma:.3f}")
    print(f"F1 (macro)         : {f1_ma:.3f}")
    print(f"Precision (weighted): {p_w:.3f}")
    print(f"Recall (weighted)   : {r_w:.3f}")
    print(f"F1 (weighted)       : {f1_w:.3f}")
    print(f"ROC-AUC (macro OvR): {auc_macro:.3f}")
    print(f"Avg Precision (macro OvR): {ap_macro:.3f}")

    # Save artifacts 
    # classification report
    cls_rep = classification_report(y_test, y_pred, digits=3)
    (REPORTS_DIR / "logreg_baseline_classification_report.txt").write_text(
        cls_rep, encoding="utf-8"
    )

    # plots
    plot_confusion(
        y_test, y_pred, REPORTS_DIR / "logreg_baseline_confusion_matrix.png"
    )
    plot_roc_ovr(
        y_test, y_proba, classes_sorted, REPORTS_DIR / "logreg_baseline_roc_ovr.png"
    )
    plot_pr_ovr(
        y_test, y_proba, classes_sorted, REPORTS_DIR / "logreg_baseline_pr_ovr.png"
    )

    print("Artifacts saved in /reports (TXT + PNGs).")

if __name__ == "__main__":
    main()
