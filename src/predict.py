from typing import List
from .config import BEST_MODEL_PATH
from .utils import load_pickle
from .preprocessing import batch_clean_text

LABEL_NAMES = {0: "NON_HATE", 1: "HATE"}

def predict_texts(texts: List[str]):
    pipe = load_pickle(BEST_MODEL_PATH)
    cleaned = batch_clean_text(texts)
    preds = pipe.predict(cleaned)
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(cleaned)
    elif hasattr(pipe, "predict_proba"):
        scores = pipe.predict_proba(cleaned)[:, 1]
    else:
        scores = [None] * len(preds)

    results = []
    for t, p, s in zip(texts, preds, scores):
        results.append({
            "text": t,
            "pred": int(p),
            "label": LABEL_NAMES.get(int(p), str(p)),
            "score": None if s is None else float(s)
        })
    return results
