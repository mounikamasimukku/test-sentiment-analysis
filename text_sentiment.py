# app.py
import streamlit as st
from PIL import Image
import pytesseract
from text_sentiment_model import load_trained_model, clean_text


# --- Set Tesseract path ONLY LOCALLY (skip this on Render) ---
import os
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# --- Check Tesseract availability ---
try:
    version = pytesseract.get_tesseract_version()
    st.write(f"Tesseract Version: {version}")
except Exception as e:
    st.error(f"Tesseract not working: {e}")

# --- Title ---
st.title("Text Sentiment Predictor")

# Load trained model
vectorizer, clf = load_trained_model()

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
        try:
            image = Image.open(image_file)
            extracted_text = pytesseract.image_to_string(image).strip()

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
        except Exception as e:
            st.error(f"❌ Error during OCR or prediction: {e}")

    if not predicted:
        st.info("Please enter a message or upload an image to analyze sentiment.")
