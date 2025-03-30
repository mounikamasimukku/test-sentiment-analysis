# Base image with Python
FROM python:3.10-slim

# Install system dependencies (Tesseract OCR)
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy local code to the container
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "text_sentiment.py", "--server.port=8501", "--server.address=0.0.0.0"]
