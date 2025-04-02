import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from PIL import Image
import pytesseract

# --- Set Tesseract path ONLY LOCALLY (skip this on Render) ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  
# --- Check Tesseract availability ---
try:
    version = pytesseract.get_tesseract_version()
    st.write(f"Tesseract Version: {version}")
except Exception as e:
    st.error(f"Tesseract not working: {e}")

# --- Ensure necessary NLTK resources are available ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- Directories ---
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --- Title ---
st.title("Text Sentiment Predictor")

# --- Load training and validation data from extracted files ---
train_df = pd.read_csv("data/train.csv", encoding='ISO-8859-1', header=0)
val_df = pd.read_csv("data/test.csv", encoding='ISO-8859-1', header=0)

# Normalize column names
train_df.columns = [col.strip().lower() for col in train_df.columns]
val_df.columns = [col.strip().lower() for col in val_df.columns]

# Use correct text column name from dataset
TEXT_COLUMN = "sentimenttext"

# Drop rows with missing values in training set
required_cols = [col for col in ["sentiment", TEXT_COLUMN] if col in train_df.columns]
train_df.dropna(subset=required_cols, inplace=True)

# --- Map numeric sentiment to labels ---
sentiment_map = {0: "negative", 1: "positive"}
train_df = train_df[train_df['sentiment'].astype(str).str.isnumeric()]
train_df['sentiment'] = train_df['sentiment'].astype(int).map(sentiment_map)

# --- Clean text function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['clean_text'] = train_df[TEXT_COLUMN].astype(str).apply(clean_text)
val_df['clean_text'] = val_df[TEXT_COLUMN].astype(str).apply(clean_text)

# --- Model Training ---
@st.cache_resource
def load_trained_model():
    X_train = train_df['clean_text']
    y_train = train_df['sentiment']
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_tfidf, y_train)
    return vectorizer, clf

vectorizer, clf = load_trained_model()

# --- Model Evaluation ---
if 'sentiment' in val_df.columns:
    X_val = val_df['clean_text']
    y_val = val_df['sentiment'].map(sentiment_map)
    X_val_tfidf = vectorizer.transform(X_val)
    y_pred = clf.predict(X_val_tfidf)

    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    cm = confusion_matrix(y_val, y_pred, labels=["negative", "positive"])
    f1 = f1_score(y_val, y_pred, average='macro')

    with open("reports/sentiment_report.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report_df.to_string())
        f.write("\n\nMacro F1 Score: {:.3f}\n".format(f1))

    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    fig_cm.savefig("images/confusion_matrix.png")
    plt.close(fig_cm)

# --- Top Words Per Sentiment ---
from collections import Counter
stop_words = set(stopwords.words('english'))
top_entities = {}
for sentiment in train_df['sentiment'].unique():
    texts = train_df[train_df['sentiment'] == sentiment]['clean_text']
    words = " ".join(texts).split()
    words = [w for w in words if w not in stop_words and w.isalpha()]
    counts = Counter(words)
    top_three = counts.most_common(3)
    top_entities[sentiment] = top_three

rows = []
for sentiment, entities in top_entities.items():
    for word, count in entities:
        rows.append({"Sentiment": sentiment, "Word": word, "Count": count})
entities_df = pd.DataFrame(rows)
with open("reports/top_entities.txt", "w") as f:
    f.write("Top Three Entities per Sentiment:\n")
    f.write(entities_df.to_string())

# --- UI: Single Page Prediction ---
st.header("Sentiment Prediction")
st.markdown("Welcome, intrepid explorer! 🌍 Ready to uncover the sentiment of your sentence?")

user_message = st.text_input("Enter a tweet or message:")
image_file = st.file_uploader("Or upload an image with text", type=['png', 'jpg', 'jpeg'])

if st.button("Predict Sentiment"):
    predicted = False

    if user_message:
        cleaned = clean_text(user_message)
        tfidf_input = vectorizer.transform([cleaned])
        prediction = clf.predict(tfidf_input)[0]
        st.success(f"**Predicted Sentiment:** {prediction}")
        predicted = True

    elif image_file is not None:
        image = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(image).strip()

        # 👇 Debug: Show OCR text
        st.write("🔍 **Extracted Text from Image:**")
        st.code(extracted_text)

        if not extracted_text:
            st.warning("No text found in the image. Please upload an image with readable text.")
        else:
            cleaned_img_text = clean_text(extracted_text)
            tfidf_input = vectorizer.transform([cleaned_img_text])
            prediction = clf.predict(tfidf_input)[0]
            st.success(f"**Predicted Sentiment from Image:** {prediction}")
            predicted = True

    if not predicted:
        st.info("Please enter a message or upload an image to analyze sentiment.")
