# models/sentiment_analysis.py
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import streamlit as st
import torch
from transformers import pipeline  # Added for BERT

# Use a smaller, faster model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# Cache and load sentiment analysis pipeline only once
@st.cache_resource(show_spinner="Loading sentiment analysis model...")
def load_sentiment_pipeline_old():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

        # Optional: log device info for debugging
        st.write(f"Sentiment model loaded on: {device}")
        return sentiment_pipe
    except Exception as e:
        st.error(f"Failed to load sentiment analysis pipeline: {e}")
        return None


@st.cache_resource(show_spinner="Loading sentiment analysis model...")
def load_sentiment_pipeline():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1)


# Manual batching for better control over performance
def get_sentiment_scores_old(texts, batch_size=32):
    if not texts or not isinstance(texts, list):
        return []

    sentiment_pipeline = load_sentiment_pipeline()
    if sentiment_pipeline is None:
        st.warning("Sentiment analysis pipeline not available.")
        return [0] * len(texts)

    # Sanitize input: replace None or empty strings
    cleaned_texts = [text if isinstance(text, str) and text.strip() else "neutral" for text in texts]

    results = []
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        try:
            results.extend(sentiment_pipeline(batch, truncation=True, max_length=512))
        except Exception as e:
            st.warning(f"Error during sentiment prediction: {e}")
            results.extend([{"label": "NEGATIVE", "score": 0.0}] * len(batch))

    # Map labels to polarity: -1 (negative), 1 (positive)
    polarities = []
    for result in results:
        label = result.get("label", "")
        if label.upper() == "POSITIVE":
            polarities.append(1)
        elif label.upper() == "NEGATIVE":
            polarities.append(-1)
        else:
            polarities.append(0)

    return polarities


def get_sentiment_scores(texts, sentiment_pipeline):
    if not texts:  # Handle empty list
        return [0] * len(texts)
    # Process texts in batches
    results = sentiment_pipeline(texts, batch_size=32, truncation=True, max_length=512)
    polarities = []
    for result in results:
        score = int(result["label"].split()[0])  # e.g., "3 stars" -> 3
        if score <= 2:
            polarities.append(-1)  # Negative
        elif score == 3:
            polarities.append(0)  # Neutral
        else:
            polarities.append(1)  # Positive
    return polarities

# def analyze_sentiment(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = softmax(outputs.logits, dim=1).numpy()[0]
#     sentiment = np.argmax(probs)
#     return sentiment, probs  # 0: Negative, 1: Neutral, 2: Positive


# def compute_bert_polarity(texts):
#     if load_bert_model is None:
#         st.warning("BERT sentiment analyzer not available. Returning zeros for polarity.")
#         return np.zeros(len(texts))
#
#     # Handle empty or invalid texts
#     texts = [str(text) if not pd.isna(text) and text.strip() != "" else "neutral" for text in texts]
#
#     # Compute sentiment scores using BERT
#     results = load_bert_model(texts)
#
#     # Map sentiment labels and scores to polarity (-1 to 1)
#     polarity_scores = []
#     for result in results:
#         score = result['score']
#         if result['label'] == 'POSITIVE':
#             polarity = score  # Positive sentiment: 0 to 1
#         else:  # NEGATIVE
#             polarity = -score  # Negative sentiment: -1 to 0
#         polarity_scores.append(polarity)
#
#     return np.array(polarity_scores)
