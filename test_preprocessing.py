# test_preprocessing.py
import pandas as pd
from src.preprocessing import batch_clean_text
from src.config import DATA_RAW

# Load dataset
df = pd.read_csv(DATA_RAW)

# Detect text column automatically
possible_cols = ["text", "comment", "sentence", "content"]
text_col = None
for c in df.columns:
    if c.lower() in possible_cols:
        text_col = c
        break

if not text_col:
    raise ValueError(f"Could not find a text column in CSV. Columns = {df.columns.tolist()}")

print("=== Preprocessing Demo (10 random rows) ===\n")

# Sample 10 random rows
samples = df.sample(10, random_state=42)

for i, row in samples.iterrows():
    original = str(row[text_col])
    cleaned = batch_clean_text([original])[0]

    print(f"Row {i}")
    print("Original :", original)
    print("Cleaned  :", cleaned)
    if "label" in df.columns:
        print("Label    :", row["label"])
    print("-" * 60)
