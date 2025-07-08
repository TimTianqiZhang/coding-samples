BRICS Discourse Classifier (BERT-Based)
🧠 Objective
Fine-tune a BERT model to classify 90 debate speech from BRICS countries’ UN General Assembly statements by:

Issue discussed (e.g., climate, economy)

Framing used (e.g., opportunity, threat)

Blame Target (e.g., global North, institutions)

Tone level (scale 1–5)

🛠️ Methods
Preprocessed 10 years of Brazil UNGA speeches into paragraph-level units

Heuristically tagged each paragraph with labels

Encoded text using BERT tokenizer

Built a multi-task BERT model with 4 output heads

Trained using PyTorch with weighted loss functions

Evaluated each output using F1-score and RMSE

📊 Result
Issue classifier: F1-score = 0.78

Framing classifier: F1-score = 0.75

Blame Target classifier: F1-score = 0.70

Tone regressor: RMSE = 0.43

Most errors stemmed from ambiguous framing or multi-issue paragraphs.

Model generalizes well to unseen BRICS country speeches (India pilot underway).

💡 Tools Used
Python, PyTorch, Hugging Face Transformers, Scikit-learn, Pandas, Matplotlib

📁 Structure
scripts/: Training and inference scripts

data/: Cleaned and labeled CSV files

models/: Saved PyTorch model weights and vectorizer

output/: Predictions and evaluation reports

▶️ How to Run

pip install -r requirements.txt
python scripts/train_multitask_bert.py
