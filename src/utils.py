import pandas as pd
import joblib
from pathlib import Path
from .config import TEXT_COL, LABEL_COL

TEXT_CANDIDATES = ["text", "comment", "sentence", "content", "message"]
LABEL_CANDIDATES = ["label", "labels", "class", "category", "target", "tag"]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _find_col(cols, candidates):
    low = [c.strip().lower() for c in cols]
    for cand in candidates:
        if cand in low:
            return cols[low.index(cand)]
    return None

def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Auto-detect delimiter; handle BOM
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

    if df.empty or len(df) == 0:
        raise ValueError(f"No rows found in CSV at {csv_path}. Check delimiter/encoding.")

    # Detect columns if names differ
    text_col = TEXT_COL if TEXT_COL in df.columns else _find_col(df.columns, TEXT_CANDIDATES)
    label_col = LABEL_COL if LABEL_COL in df.columns else _find_col(df.columns, LABEL_CANDIDATES)

    if text_col is None or label_col is None:
        raise ValueError(
            "Could not find text/label columns. "
            f"Found columns={list(df.columns)}. "
            "Expected something like text/comment/sentence and label/class/category."
        )

    # Clean up text & drop empties
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].copy()

    # Map labels to 0/1 robustly
    # Map labels to 0/1 robustly (handles "Hate-Speech" / "Non-Hate-Speech")
    if df[label_col].dtype == object:
        lab = df[label_col].astype(str).str.strip().str.lower()

        def map_label(x: str) -> int:
            # normalize: remove spaces, hyphens, underscores
            x = x.replace(" ", "").replace("-", "").replace("_", "")
            # explicit negatives FIRST
            if x in {"nonhatespeech", "nonhate", "nothate"}:
                return 0
            # explicit positives
            if x in {"hatespeech", "hate"}:
                return 1
            # fallbacks
            if ("non" in x and "hate" in x) or ("no" in x and "hate" in x):
                return 0
            if "hate" in x:
                return 1
            return 0  # default to non-hate

        df[label_col] = lab.map(map_label)

    # convert to numeric 0/1, drop bad rows, and enforce binary
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    df = df[df[label_col].isin([0, 1])].copy()


    # keep binary only
    df = df[df[label_col].isin([0, 1])].copy()

    if len(df) == 0:
        raise ValueError(
            "After cleaning, no valid samples remained. "
            "Check that your label column contains 0/1 or recognizable strings (HATE/NON_HATE, etc.)."
        )

    # Rename to expected names
    if text_col != TEXT_COL:
        df = df.rename(columns={text_col: TEXT_COL})
    if label_col != LABEL_COL:
        df = df.rename(columns={label_col: LABEL_COL})

    return df

def save_pickle(obj, path: Path):
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_pickle(path: Path):
    return joblib.load(path)
