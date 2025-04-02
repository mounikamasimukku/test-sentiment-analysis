# model.py
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

# --- Load and train the model ---
def load_trained_model():
    train_df = pd.read_csv("data/train.csv", encoding='ISO-8859-1', header=0)
    train_df.columns = [col.strip().lower() for col in train_df.columns]
    TEXT_COLUMN = "sentimenttext"
    train_df.dropna(subset=['sentiment', TEXT_COLUMN], inplace=True)

    sentiment_map = {0: "negative", 1: "positive"}
    train_df = train_df[train_df['sentiment'].astype(str).str.isnumeric()]
    train_df['sentiment'] = train_df['sentiment'].astype(int).map(sentiment_map)

    train_df['clean_text'] = train_df[TEXT_COLUMN].astype(str).apply(clean_text)

    X_train = train_df['clean_text']
    y_train = train_df['sentiment']
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_tfidf, y_train)

    return vectorizer, clf
