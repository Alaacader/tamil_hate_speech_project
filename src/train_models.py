from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from .config import (
    DATA_RAW,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    N_JOBS,
    BEST_MODEL_PATH,
    TEXT_COL,
    LABEL_COL,
)
from .utils import load_dataset, save_pickle
from .preprocessing_train import batch_clean_text
from .feature_extraction import build_tfidf_vectorizer, build_word_tfidf_vectorizer


def build_pipeline(clf):
    # Combine char + word tfidf using FeatureUnion
    char_vec = ("char_tfidf", build_tfidf_vectorizer())
    word_vec = ("word_tfidf", build_word_tfidf_vectorizer())

    union = FeatureUnion([char_vec, word_vec])

    pipe = Pipeline([
        ("features", union),
        ("clf", clf)
    ])
    return pipe


def get_model_spaces() -> Dict[str, Dict[str, Any]]:
    return {
        "logreg": {
            "estimator": LogisticRegression(max_iter=200, n_jobs=N_JOBS),
            "param_grid": {
                "clf__C": [0.5, 1.0, 2.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "linear_svm": {
            "estimator": LinearSVC(),
            "param_grid": {
                "clf__C": [0.5, 1.0, 2.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "mnb": {
            "estimator": MultinomialNB(),
            "param_grid": {
                "clf__alpha": [0.1, 0.5, 1.0],
            },
        },
        "rf": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            "param_grid": {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 20, 40],
            },
        },
    }


def train_and_select_best():
    df = load_dataset(DATA_RAW)

    # --- D) Debug: dataset shape and label counts ---
    from collections import Counter
    print("[DEBUG] df shape:", df.shape)
    print("[DEBUG] LABEL_COL:", LABEL_COL, "| TEXT_COL:", TEXT_COL)
    print("[DEBUG] raw label counts:", Counter(df[LABEL_COL].astype(str)))
    try:
        y_debug = df[LABEL_COL].astype(int)
        print("[DEBUG] numeric label counts:", Counter(y_debug))
    except Exception as e:
        print("[DEBUG] numeric label counts: (not numeric yet)", e)

    # --- Preprocess text ---
    X = batch_clean_text(df[TEXT_COL].astype(str).tolist())
    y = df[LABEL_COL].values

    # --- E) Debug: empty after preprocessing ---
    empty = sum(1 for x in X if not x.strip())
    print(f"[DEBUG] empty texts after preprocessing: {empty} / {len(X)}")

    # ---- Robust split: only stratify if both classes exist AND each has >= 2 samples ----
    unique_classes, counts = np.unique(y, return_counts=True)
    can_stratify = (len(unique_classes) >= 2) and (counts.min() >= 2)

    strat = y if can_stratify else None
    if not can_stratify:
        print("[WARN] Not enough samples per class for stratify; proceeding without stratify.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )
    # -------------------------------------------------------------------------------

    results = []
    best_model = None
    best_score = -np.inf
    best_name = None

    for name, space in get_model_spaces().items():
        estimator = space["estimator"]
        param_grid = space["param_grid"]

        pipe = build_pipeline(estimator)
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=CV_FOLDS,
            n_jobs=N_JOBS,
            scoring="f1"
        )
        gs.fit(X_train, y_train)

        val_pred = gs.best_estimator_.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        results.append((name, gs.best_score_, val_acc, gs.best_params_))
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_
            best_name = name

    from pathlib import Path
    print("[DEBUG] Saving best model to:", BEST_MODEL_PATH)
    Path(BEST_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Save best pipeline
    save_pickle(best_model, BEST_MODEL_PATH)
    return best_name, results

