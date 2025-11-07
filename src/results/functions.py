import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

def load_model(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
    else:
        raise ValueError("invalid format")
    return model


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted"),
        "recall": recall_score(y, y_pred, average="weighted"),
        "f1-score": f1_score(y, y_pred, average="weighted"),
    }
    return y_pred, metrics


def confusionMatrix(y_true, y_pred, model_name):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    fig.suptitle(model_name, fontsize=16, fontweight="semibold", y=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High", "Critical"])
    cm.plot(cmap="Blues", ax=ax[0], colorbar=False)
    ax[0].set_title("Confusion Matrix", fontsize=16)
    ax[0].grid(False)

    # Classification report
    ax[1].text(
        x=0.5,
        y=0.5,
        s=classification_report(y_true, y_pred, target_names=["Low", "Medium", "High", "Critical"]),
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="monospace",
    )
    ax[1].set_title("Classification Report", fontsize=16)
    ax[1].axis("off")

    plt.tight_layout()
    