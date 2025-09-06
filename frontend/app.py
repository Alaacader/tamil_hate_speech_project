# frontend/app.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import time
import math
from pathlib import Path

import streamlit as st
from deep_translator import GoogleTranslator

from src.config import BEST_MODEL_PATH
from src.utils import load_pickle
from src.preprocessing import batch_clean_text

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Tamil Hate Speech Detector", page_icon="🛡️")

st.title("🛡️ Tamil Hate Speech Detector")
st.caption(
    "Paste Tamil text below. The app predicts **HATE** or **NON_HATE**, shows a "
    "**confidence %**, and an **English translation**. No inputs are stored."
)

# -------------------------
# Helpers
# -------------------------
TA_REGEX = re.compile(r"[\u0B80-\u0BFF]")  # Tamil Unicode block

def looks_tamil(text: str) -> bool:
    """Heuristic: True if the text contains at least one Tamil char."""
    return bool(TA_REGEX.search(text or ""))

def score_to_confidence(score):
    """Turn decision_function or probability into 0–100%."""
    if score is None:
        return None
    if 0.0 <= score <= 1.0:
        return round(score * 100, 1)  # probability
    return round(100 / (1 + math.exp(-score)), 1)  # logistic for decision_function

# Ensure session key exists BEFORE creating the widget
if "ta_input" not in st.session_state:
    st.session_state.ta_input = ""

def clear_text():
    """Button callback to safely clear textarea."""
    st.session_state.ta_input = ""
    try:
        st.rerun()              # Streamlit >= 1.29
    except Exception:
        st.experimental_rerun() # older versions

# -------------------------
# Load model
# -------------------------
model_path: Path = BEST_MODEL_PATH
if not model_path.exists():
    st.error("Model not found. Please train it first: `python run.py --train`.")
    st.stop()

pipe = load_pickle(model_path)

# -------------------------
# UI
# -------------------------
user_text = st.text_area("Paste Tamil text", height=150, key="ta_input")

c1, c2 = st.columns([1, 1])
classify_clicked = c1.button("Classify")
c2.button("Clear", on_click=clear_text)

if not classify_clicked:
    st.info("Enter some Tamil text and click **Classify**.")
    st.caption("Predictions are automated and may be imperfect.")
    st.stop()

# -------------------------
# Classification flow
# -------------------------
text = (user_text or "").strip()
if not text:
    st.error("Please enter Tamil text.")
    st.stop()

if not looks_tamil(text):
    st.warning("The text does not appear to contain Tamil characters. Results may be unreliable.")

# Time the prediction
t0 = time.perf_counter()
cleaned = batch_clean_text([text])

pred = pipe.predict(cleaned)[0]
label = "HATE" if int(pred) == 1 else "NON_HATE"

score = None
if hasattr(pipe, "decision_function"):
    try:
        score = float(pipe.decision_function(cleaned)[0])
    except Exception:
        score = None
elif hasattr(pipe, "predict_proba"):
    try:
        score = float(pipe.predict_proba(cleaned)[0][1])
    except Exception:
        score = None

elapsed_ms = (time.perf_counter() - t0) * 1000.0
confidence = score_to_confidence(score)

# Results
# Color accent for label
if label == "HATE":
    st.markdown("### Prediction: :red[HATE]")
else:
    st.markdown("### Prediction: :green[NON_HATE]")

if confidence is not None:
    st.caption(f"Confidence: **{confidence:.1f}%**")
st.caption(f"Time to predict: **{elapsed_ms:.0f} ms**")
st.caption("Model: Logistic Regression (TF-IDF)")  # Update if you switch models

# Translation (always on)
st.markdown("**English translation:**")
try:
    translation = GoogleTranslator(source="ta", target="en").translate(text)
    st.write(translation)
except Exception as e:
    st.caption(f"Translation unavailable right now ({e}).")

st.divider()
st.caption("© Final Year Project — Tamil Hate Speech Detection | No inputs are stored.")
