import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.metrics import f1_score
import config

def load_data(base_path="data"):
    train_df = pd.read_csv(f"{base_path}/train.csv")
    valid_df = pd.read_csv(f"{base_path}/valid.csv")
    test_df = pd.read_csv(f"{base_path}/test.csv")

    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}

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


def create_text_vectorizer(
    train_texts,
    max_tokens=config.MAX_TOKENS,
    output_sequence_length=config.OUTPUT_SEQ_LEN,
):
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_sequence_length,
        name="text_vectorizer_lstm",
    )
    vectorizer.adapt(train_texts)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    return vectorizer, vocab, vocab_size


def compute_f1_score(y_true, y_pred_probs):
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    return f1_score(y_true, y_pred_classes, average="macro", zero_division=0)
