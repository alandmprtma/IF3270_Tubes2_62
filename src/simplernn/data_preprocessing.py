import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import f1_score, classification_report
import config


# Load datasets
def load_data():
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")
    test_df = pd.read_csv("data/test.csv")

    # Map text labels to integer indices
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}

    # Convert labels
    train_labels = np.array([label_mapping[label] for label in train_df["label"]])
    valid_labels = np.array([label_mapping[label] for label in valid_df["label"]])
    test_labels = np.array([label_mapping[label] for label in test_df["label"]])

    return (
        (train_df["text"].values, train_labels),
        (valid_df["text"].values, valid_labels),
        (test_df["text"].values, test_labels),
        label_mapping,
        len(label_mapping),
    )


# Create text vectorization layer for tokenization
def create_text_vectorizer(train_texts, max_tokens=config.MAX_TOKENS, output_sequence_length=config.OUTPUT_SEQ_LEN):
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_sequence_length,
        name="text_vectorizer",
    )

    # Adapt the vectorizer to the training text
    vectorizer.adapt(train_texts)

    # Get vocabulary
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)

    return vectorizer, vocab, vocab_size


# Compute macro F1 score for final evaluation
def compute_f1_score(y_true, y_pred):
    y_pred_classes = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred_classes, average="macro")
