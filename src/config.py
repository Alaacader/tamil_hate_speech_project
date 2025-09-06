from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "tamil_hate_speech.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_pipeline.pkl"

# Column names in your CSV
TEXT_COL = "text"
LABEL_COL = "label"

# Train/validation split
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1  # parallel jobs for scikit-learn

# Cross-validation
CV_FOLDS = 5

# Label mapping (if labels are strings)
LABEL_MAPPING = {
    "HATE": 1,
    "NON_HATE": 0,
    "hate": 1,
    "non_hate": 0,
    "Hate": 1,
    "Non_Hate": 0,
}
