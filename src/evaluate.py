import matplotlib
matplotlib.use("Agg")   # use non-GUI backend to avoid Tkinter errors

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import inspect
from sklearn.metrics import classification_report, confusion_matrix

from .config import DATA_RAW, BEST_MODEL_PATH, TEXT_COL, LABEL_COL
from .utils import load_dataset, load_pickle
from .preprocessing_train import batch_clean_text
print("[DEBUG] evaluate.py using preprocessor:", batch_clean_text.__module__)


def evaluate_saved_model():
    df = load_dataset(DATA_RAW)
    # After: df = load_dataset(DATA_RAW)
    from collections import Counter

    print("[DEBUG] df shape:", df.shape)
    print("[DEBUG] LABEL_COL:", LABEL_COL, "| TEXT_COL:", TEXT_COL)

    # Show raw label distribution BEFORE any mapping (if you map elsewhere, move this after mapping)
    print("[DEBUG] raw label counts:", Counter(df[LABEL_COL].astype(str)))

    # If your pipeline already maps labels to ints (0/1), print numeric counts too:
    try:
        y_debug = df[LABEL_COL].astype(int)
        print("[DEBUG] numeric label counts:", Counter(y_debug))
    except Exception as e:
        print("[DEBUG] numeric label counts: (not numeric yet)", e)

    X = batch_clean_text(df[TEXT_COL].astype(str).tolist())
    y_true = df[LABEL_COL].values

    from pathlib import Path, PurePath
    print("[DEBUG] BEST_MODEL_PATH =", BEST_MODEL_PATH)
    p = Path(BEST_MODEL_PATH)
    print("[DEBUG] Model exists?", p.exists(), "mtime:", p.stat().st_mtime if p.exists() else "NA")

    pipe = load_pickle(BEST_MODEL_PATH)
    y_pred = pipe.predict(X)

    # Print classification report to console
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    # Save confusion matrix (raw counts)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.title('Confusion Matrix (Entire Dataset)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[INFO] Confusion matrix saved to confusion_matrix.png")

    # Save classification report to CSV
    report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
    pd.DataFrame(report_dict).to_csv("classification_report.csv")
    print("[INFO] Classification report saved to classification_report.csv")

    # Save normalized confusion matrix (percentages)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cbar=False)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig("confusion_matrix_normalized.png")
    print("[INFO] Normalized confusion matrix saved to confusion_matrix_normalized.png")
