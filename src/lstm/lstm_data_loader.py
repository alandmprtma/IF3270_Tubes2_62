import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np

# --- Configuration ---
DATA_PATH = "data/" # Relative to the lstm folder
TRAIN_FILE = DATA_PATH + "train.csv"
VALID_FILE = DATA_PATH + "valid.csv"
TEST_FILE = DATA_PATH + "test.csv"

# TextVectorization parameters
MAX_TOKENS = 10000  # Vocabulary size
OUTPUT_SEQUENCE_LENGTH = 250  # Max length of sequences (adjust as needed)

# Define a mapping for labels
label_map = {"positive": 0, "neutral": 1, "negative": 2}
# Inverse map for later, if needed for interpretation
inverse_label_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# --- 1. Load Data ---
def load_dataframe(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns.")
        df['text'] = df['text'].fillna('')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# --- 2. Prepare Labels ---
def encode_labels(labels_series):
    """Encodes string labels to integers."""
    return labels_series.map(label_map)

def get_processed_data():
    """Loads, preprocesses, and returns the data."""
    print("--- Starting Data Loading and Preprocessing ---")

    train_df = load_dataframe(TRAIN_FILE)
    valid_df = load_dataframe(VALID_FILE)
    test_df = load_dataframe(TEST_FILE)

    if train_df is None or valid_df is None or test_df is None:
        print("Aborting due to data loading errors.")
        return None, None, None, None, None, None, None, None

    train_texts = train_df['text'].astype(str).tolist()
    train_labels_str = train_df['label'].tolist()
    valid_texts = valid_df['text'].astype(str).tolist()
    test_texts = test_df['text'].astype(str).tolist()

    train_labels = encode_labels(train_df['label'])
    valid_labels = encode_labels(valid_df['label'])
    test_labels = encode_labels(test_df['label'])

    if train_labels.isnull().any() or valid_labels.isnull().any() or test_labels.isnull().any():
        print("Error: Found NaN values in labels after mapping. Check your label_map and CSV files.")
        return None, None, None, None, None, None, None, None
    
    train_labels = train_labels.to_numpy()
    valid_labels = valid_labels.to_numpy()
    test_labels = test_labels.to_numpy()

    print(f"\nNumber of training samples: {len(train_texts)}")
    print(f"Number of validation samples: {len(valid_texts)}")
    print(f"Number of test samples: {len(test_texts)}")
    print(f"First training text: '{train_texts[0]}'")
    print(f"Its original label: '{train_labels_str[0]}', Encoded label: {train_labels[0]}")

    print("\n--- Initializing and Adapting TextVectorization Layer ---")
    text_vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    )
    print("Adapting TextVectorization to training data...")
    text_vectorizer.adapt(train_texts)
    print("Adaptation complete.")

    vocab = text_vectorizer.get_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    # print(f"First 10 tokens in vocab: {vocab[:10]}")

    print("\nTokenizing text data...")
    x_train_tokens = text_vectorizer(np.array(train_texts))
    x_valid_tokens = text_vectorizer(np.array(valid_texts))
    x_test_tokens = text_vectorizer(np.array(test_texts))

    print(f"Shape of tokenized training data: {x_train_tokens.shape}")
    # print(f"Original text (train[0]): '{train_texts[0]}'")
    # print(f"Tokenized sequence (train[0]): {x_train_tokens[0][:20]}...")

    print("\n--- Data Loading and Preprocessing Complete ---")
    
    return (x_train_tokens, train_labels,
            x_valid_tokens, valid_labels,
            x_test_tokens, test_labels,
            text_vectorizer, num_classes)

if __name__ == '__main__':
    # This part now just demonstrates how to use the function
    # and allows you to run the script directly to test data loading.
    processed_data = get_processed_data()
    if processed_data and all(item is not None for item in processed_data):
        (x_train, y_train, x_val, y_val, x_test, y_test, vectorizer, n_classes) = processed_data
        print(f"\nSuccessfully retrieved processed data.")
        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        print(f"Number of classes: {n_classes}")
        print(f"Vocabulary size from vectorizer: {len(vectorizer.get_vocabulary())}")
    else:
        print("Failed to retrieve processed data.")