# models/sentiment_analysis.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
import numpy as np
import streamlit as st


@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=3)  # Positive, Negative, Neutral
    return tokenizer, model


def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1).numpy()[0]
    sentiment = np.argmax(probs)
    return sentiment, probs  # 0: Negative, 1: Neutral, 2: Positive
