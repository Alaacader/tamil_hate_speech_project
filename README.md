# Tamil Hate Speech Detection (PyCharm project)

## 1) Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

## 2) Install dependencies
pip install -r requirements.txt

## 3) Add your dataset
Place a CSV at: data/raw/tamil_hate_speech.csv

Required columns:
- text  : sentence/comment in Tamil
- label : 0 or 1 (or HATE / NON_HATE)

## 4) Train models
python run.py --train

## 5) Evaluate
python run.py --evaluate

## 6) Predict on a quick sample
python run.py --predict --text "இந்த கருத்து வெறுப்பை பரப்புகிறது"

## 7) Launch the Streamlit app
streamlit run frontend/app.py
