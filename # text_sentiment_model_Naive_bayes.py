# text_sentiment_model.py
import pandas as pd
import numpy as np
import os
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# --- Clean text function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Load and train model (cached) ---
@st.cache_resource
def load_trained_model():
    # Load dataset
    train_df = pd.read_csv("data/train.csv", encoding='ISO-8859-1')
    val_df = pd.read_csv("data/test.csv", encoding='ISO-8859-1')

    # Normalize column names
    train_df.columns = [col.strip().lower() for col in train_df.columns]
    val_df.columns = [col.strip().lower() for col in val_df.columns]

    TEXT_COLUMN = "sentimenttext"

    # Drop missing
    required_cols = [col for col in ["sentiment", TEXT_COLUMN] if col in train_df.columns]
    train_df.dropna(subset=required_cols, inplace=True)

    sentiment_map = {0: "negative", 1: "positive"}
    train_df = train_df[train_df['sentiment'].astype(str).str.isnumeric()]
    train_df['sentiment'] = train_df['sentiment'].astype(int).map(sentiment_map)

    train_df['clean_text'] = train_df[TEXT_COLUMN].astype(str).apply(clean_text)

    X_train = train_df['clean_text']
    y_train = train_df['sentiment']

    # TF-IDF with fewer features
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    return vectorizer, clf
