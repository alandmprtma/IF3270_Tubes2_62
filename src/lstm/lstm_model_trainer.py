import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers # Import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score  # Import F1 score
import numpy as np  # For argmax
from lstm_data_loader import (
    get_processed_data,
    OUTPUT_SEQUENCE_LENGTH,
    num_classes,
)  # MAX_TOKENS is not directly needed here if vocab_size is from vectorizer


# --- 1. Model Definition ---
# ... (build_lstm_model function remains the same as your provided version) ...
def build_lstm_model(vocab_size, embedding_dim, lstm_units, num_classes_model, bidirectional=False, num_lstm_layers=1, output_sequence_length=100, l2_reg=0.001): # Added l2_reg parameter
    """Builds a Keras LSTM model with L2 regularization."""
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=output_sequence_length))

    for i in range(num_lstm_layers):
        is_last_lstm_layer = (i == num_lstm_layers - 1)
        return_sequences_flag = not is_last_lstm_layer
        
        if bidirectional:
            model.add(Bidirectional(LSTM(units=lstm_units, 
                                         return_sequences=return_sequences_flag,
                                         kernel_regularizer=regularizers.l2(l2_reg))))
        else:
            model.add(LSTM(units=lstm_units, 
                           return_sequences=return_sequences_flag,
                           kernel_regularizer=regularizers.l2(l2_reg)))

    model.add(Dropout(0.5)) # Keep dropout as is for now, or slightly increase if L2 alone is not enough
    model.add(Dense(units=num_classes_model, 
                    activation='softmax',
                    kernel_regularizer=regularizers.l2(l2_reg))) # Added L2 reg
    
    model.build(input_shape=(None, output_sequence_length)) 

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 2. Training Configuration ---
EMBEDDING_DIM = 128
LSTM_UNITS = 128
NUM_LSTM_LAYERS = 1
IS_BIDIRECTIONAL = True
L2_REGULARIZATION_STRENGTH = 0.001 # Define L2 strength
EPOCHS = 25
BATCH_SIZE = 32
PATIENCE_EARLY_STOPPING = 5

# --- 3. Plotting Function ---
# ... (plot_history function remains the same as your provided version) ...
def plot_history(history):
    """Plots training & validation accuracy and loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig("lstm_training_history.png")
    print("Training history plot saved as lstm_training_history.png")


def main():
    print("--- Loading and Preprocessing Data ---")
    processed_data = get_processed_data()
    if not processed_data or not all(item is not None for item in processed_data):
        print("Failed to load data. Exiting.")
        return

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        text_vectorizer,
        n_classes_from_data,
    ) = processed_data

    if n_classes_from_data != num_classes:
        print(
            f"Warning: num_classes mismatch! From data loader: {n_classes_from_data}, Expected: {num_classes}"
        )

    vocab_size = len(text_vectorizer.get_vocabulary())
    print(f"Actual vocabulary size: {vocab_size}, Sequence length: {OUTPUT_SEQUENCE_LENGTH}, Num classes: {n_classes_from_data}")

    print("\n--- Building LSTM Model ---")
    # Parameters for build_lstm_model will now use the updated Training Configuration
    model = build_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        num_classes_model=n_classes_from_data,
        bidirectional=IS_BIDIRECTIONAL,
        num_lstm_layers=NUM_LSTM_LAYERS,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
        l2_reg=L2_REGULARIZATION_STRENGTH 
    )
    model.summary()

    print("\n--- Training LSTM Model ---")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE_EARLY_STOPPING, restore_best_weights=True
    )
    model_checkpoint_path = "lstm_model_best.keras"
    model_checkpoint = ModelCheckpoint(
        model_checkpoint_path, save_best_only=True, monitor="val_loss"
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
    )

    print("\n--- Evaluating Model ---")
    best_model = tf.keras.models.load_model(model_checkpoint_path)

    loss, accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_probs = best_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    macro_f1 = f1_score(y_test, y_pred_classes, average="macro")
    print(f"Test Macro F1-Score: {macro_f1:.4f}")

    print(f"Best model saved to {model_checkpoint_path}")

    print("\n--- Plotting Training History ---")
    plot_history(history)  # Ensure this line is present


if __name__ == "__main__":
    main()
