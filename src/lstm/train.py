import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import time
import config
from data_preprocessing import load_data, create_text_vectorizer, compute_f1_score
from model import create_lstm_model


def train_and_evaluate_lstm_model(
    num_lstm_layers=config.NUM_LSTM_LAYERS,
    lstm_units=config.LSTM_UNITS,
    bidirectional=config.BIDIRECTIONAL_LSTM,
    dropout_rate=config.DROPOUT_RATE,
    embedding_dropout_rate=config.EMBEDDING_DROPOUT,
    recurrent_dropout_rate=config.RECURRENT_DROPOUT_LSTM,
    l2_reg=config.L2_REG,
    learning_rate=config.LEARNING_RATE,
    model_name="lstm_model",
    use_class_weights=True,
):
    print(f"\n--- Training LSTM Model: {model_name} ---")
    print(
        f"Config: Layers={num_lstm_layers}, Units={lstm_units}, "
        f"Bidirectional={bidirectional}, Dropout={dropout_rate}, EmbDropout={embedding_dropout_rate}, RecDropout={recurrent_dropout_rate}, L2={l2_reg}"
    )

    # Muat dataset
    (
        (train_texts, train_labels),
        (valid_texts, valid_labels),
        (test_texts, test_labels),
        label_mapping,
        num_classes,
    ) = load_data()  

    print(
        f"Ukuran dataset: Train={len(train_texts)}, Valid={len(valid_texts)}, Test={len(test_texts)}"
    )
    print(f"Distribusi kelas di data train: {np.bincount(train_labels)}")

    # Hitung bobot kelas untuk mengatasi ketidakseimbangan data
    if use_class_weights:
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        # Hindari pembagian dengan nol jika ada kelas yang tidak ada di train_labels (jarang terjadi tapi aman)
        class_weights = {
            i: total_samples / (len(class_counts) * count) if count > 0 else 0
            for i, count in enumerate(class_counts)
        }
        print(f"Menggunakan bobot kelas: {class_weights}")
    else:
        class_weights = None

    # Buat dan siapkan text vectorizer
    vectorizer, vocab, vocab_size = create_text_vectorizer(
        train_texts,
        max_tokens=config.MAX_TOKENS,
        output_sequence_length=config.OUTPUT_SEQ_LEN,
    )
    print(f"Ukuran vocabulary: {vocab_size}")

    # Ubah teks jadi sequence angka
    train_sequences = vectorizer(train_texts)
    valid_sequences = vectorizer(valid_texts)
    test_sequences = vectorizer(test_texts)

    # Buat dataset TF untuk efisiensi training
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
    train_dataset = (
        train_dataset.shuffle(
            len(train_texts) * 4, reshuffle_each_iteration=True, seed=config.RANDOM_SEED
        )
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_sequences, valid_labels))
    valid_dataset = valid_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
    test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Buat model LSTM sesuai konfigurasi
    model = create_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=lstm_units,
        num_lstm_layers=num_lstm_layers,
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
        embedding_dropout_rate=embedding_dropout_rate,
        recurrent_dropout_rate=recurrent_dropout_rate,
        l2_reg_strength=l2_reg,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    # Tampilkan ringkasan model
    model.summary()

    os.makedirs("checkpoints", exist_ok=True)
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

    # Latih model
    print("Mulai pelatihan model LSTM...")
    start_train_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
    )
    training_duration = time.time() - start_train_time
    print(f"Pelatihan selesai dalam {training_duration:.2f} detik.")

    # Evaluasi di test set (model sudah restore bobot terbaik jika EarlyStopping digunakan)
    print("Evaluasi di test set dengan bobot terbaik...")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)

    # Ambil prediksi untuk menghitung F1
    test_pred_probs = model.predict(test_dataset)
    test_f1 = compute_f1_score(
        test_labels, test_pred_probs
    )  # Menggunakan fungsi dari data_preprocessing

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    # Hitung dan tampilkan metrik per kelas
    y_pred_classes = np.argmax(test_pred_probs, axis=1)
    from sklearn.metrics import (
        classification_report,
    )

    class_names = list(label_mapping.keys())
    print("\nLaporan Klasifikasi:")
    print(
        classification_report(
            test_labels, y_pred_classes, target_names=class_names, zero_division=0
        )
    )

    os.makedirs("models", exist_ok=True)
    full_model_path = f"models/{model_name}_full_model.keras"
    model.save(full_model_path)
    print(f"Model lengkap disimpan di: {full_model_path}")

    # Simpan vectorizer
    vectorizer_path = f"models/{model_name}_vectorizer.keras"
    # Buat model Sequential sementara hanya untuk menyimpan TextVectorization layer
    vectorizer_export_model = tf.keras.Sequential(
        [vectorizer], name="text_vectorization_model_lstm"
    )
    vectorizer_export_model.save(vectorizer_path, save_format="keras")
    print(f"Vectorizer disimpan di: {vectorizer_path}")

    return (
        model,
        history,
        test_pred_probs,
        test_labels,
        vectorizer,
        {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "training_time": training_duration,
        },
    )


def plot_training_history(history, model_name):
    """Plot dan simpan kurva history training"""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Loss Training")
    plt.plot(history.history["val_loss"], label="Loss Validasi")
    plt.title(f"Kurva Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Akurasi Training")
    plt.plot(history.history["val_accuracy"], label="Akurasi Validasi")
    plt.title(f"Kurva Akurasi - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Contoh penggunaan (bisa dijalankan jika file ini dieksekusi langsung)
if __name__ == "__main__":
    print("Menjalankan contoh pelatihan model LSTM default...")

    # Pastikan direktori output ada
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    (
        model_lstm,
        history_lstm,
        preds_lstm,
        labels_true_lstm,
        vectorizer_lstm,
        metrics_lstm,
    ) = train_and_evaluate_lstm_model(model_name="default_lstm_model")

    print("\nMetrik evaluasi model LSTM default:")
    for name, value in metrics_lstm.items():
        print(f"- {name}: {value:.4f}")

    plot_training_history(history_lstm, "default_lstm_model")
