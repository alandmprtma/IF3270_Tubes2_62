import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_data, create_text_vectorizer, compute_f1_score
from model import create_rnn_model
import config
import os

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def train_and_evaluate_model(
    num_rnn_layers=config.NUM_RNN_LAYERS,
    rnn_units=config.RNN_UNITS,
    bidirectional=config.BIDIRECTIONAL,
    dropout_rate=config.DROPOUT_RATE,
    l2_reg=config.L2_REG,
    learning_rate=config.LEARNING_RATE,
    model_name="simple_rnn_model",
    use_class_weights=True,
):
    """
    Train and evaluate a SimpleRNN model with the specified hyperparameters.
    All default values are pulled from config.py.
    """
    print(f"\n--- Training Model: {model_name} ---")
    print(
        f"Config: Layers={num_rnn_layers}, Units={rnn_units}, "
        f"Bidirectional={bidirectional}, Dropout={dropout_rate}, L2={l2_reg}"
    )

    # Load data
    (
        (train_texts, train_labels),
        (valid_texts, valid_labels),
        (test_texts, test_labels),
        label_mapping,
        num_classes,
    ) = load_data()

    print(f"Dataset sizes: Train={len(train_texts)}, Valid={len(valid_texts)}, Test={len(test_texts)}")
    print(f"Class distribution in train: {np.bincount(train_labels)}")

    # Calculate class weights if needed
    if use_class_weights:
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = {
            i: total_samples / (len(class_counts) * count)
            for i, count in enumerate(class_counts)
        }
        print(f"Using class weights: {class_weights}")
    else:
        class_weights = None

    # Create and adapt text vectorizer
    vectorizer, vocab, vocab_size = create_text_vectorizer(
        train_texts, max_tokens=config.MAX_TOKENS, output_sequence_length=config.OUTPUT_SEQ_LEN
    )
    print(f"Vocabulary size: {vocab_size}")

    # Convert texts to sequences
    train_sequences = vectorizer(train_texts)
    valid_sequences = vectorizer(valid_texts)
    test_sequences = vectorizer(test_texts)

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
    train_dataset = (
        train_dataset.shuffle(len(train_texts) * 4, reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_sequences, valid_labels))
    valid_dataset = valid_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
    test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create model
    model = create_rnn_model(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        rnn_units=rnn_units,
        num_rnn_layers=num_rnn_layers,
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    # Display model summary
    model.summary()

    # Set up callbacks
    checkpoint_path = f"checkpoints/{model_name}.weights.h5"
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.ES_PATIENCE,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
    ]

    # Train model
    print("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
    )

    # Evaluate on test set
    print("Evaluating on test set with best weights...")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)

    # Get predictions for F1 calculation
    test_pred_probs = model.predict(test_dataset)
    test_f1 = compute_f1_score(test_labels, test_pred_probs)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    # Calculate and print metrics per class
    y_pred = np.argmax(test_pred_probs, axis=1)
    from sklearn.metrics import classification_report, confusion_matrix

    class_names = list(label_mapping.keys())
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save model
    full_model_path = f"models/{model_name}_full_model.keras"
    model.save(full_model_path)
    print(f"Full model saved to: {full_model_path}")

    # Save vectorizer
    vectorizer_path = f"models/{model_name}_vectorizer.keras"
    tf.keras.models.save_model(
        tf.keras.Sequential([vectorizer], name="text_vectorization_model"),
        vectorizer_path,
    )
    print(f"Vectorizer saved to: {vectorizer_path}")

    return (
        model,
        history,
        test_pred_probs,
        test_labels,
        vectorizer,
        {"test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1},
    )


def plot_training_history(history, model_name):
    """Plot and save training history curves"""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plot_path = f"plots/{model_name}_training_curves.png"
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Train baseline model - we can override specific parameters here
    model_results = train_and_evaluate_model(
        num_rnn_layers=1,
        rnn_units=48,
        bidirectional=True,
        model_name="simplernn_baseline",
    )
    
    model, history, _, _, _, metrics = model_results
    plot_training_history(history, "simplernn_baseline")
    
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")