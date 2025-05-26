import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_data, create_text_vectorizer, compute_f1_score
from model import create_rnn_model
import config
import os


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
    print(f"\n--- Training Model: {model_name} ---")
    print(
        f"Config: Layers={num_rnn_layers}, Units={rnn_units}, "
        f"Bidirectional={bidirectional}, Dropout={dropout_rate}, L2={l2_reg}"
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
        class_weights = {
            i: total_samples / (len(class_counts) * count)
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
        train_dataset.shuffle(len(train_texts) * 4, reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_sequences, valid_labels))
    valid_dataset = valid_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
    test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Buat model sesuai konfigurasi
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

    if config.OUTPUT_SEQ_LEN is not None:
        model.build(input_shape=(None, config.OUTPUT_SEQ_LEN))
    else:
        print("Peringatan: config.OUTPUT_SEQ_LEN tidak disetel, model.build() dilewati. Summary mungkin unbuilt.")

    # Tampilkan ringkasan model
    model.summary()

    # Siapkan callback untuk training
    checkpoint_path = f"checkpoints/{model_name}.weights.h5"
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",  # Pantau val_loss untuk simpan model terbaik
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.ES_PATIENCE,  # Berhenti kalau tidak ada perbaikan
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.LR_FACTOR,  # Faktor pengurangan learning rate
            patience=config.LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
    ]

    # Latih model
    print("Mulai pelatihan model...")
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
    )

    # Evaluasi di test set
    print("Evaluasi di test set dengan bobot terbaik...")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)

    # Ambil prediksi untuk menghitung F1
    test_pred_probs = model.predict(test_dataset)
    test_f1 = compute_f1_score(test_labels, test_pred_probs)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    # Hitung dan tampilkan metrik per kelas
    y_pred = np.argmax(test_pred_probs, axis=1)
    from sklearn.metrics import classification_report, confusion_matrix

    class_names = list(label_mapping.keys())
    print("\nLaporan Klasifikasi:")
    print(
        classification_report(
            test_labels, y_pred, target_names=class_names, zero_division=0
        )
    )

    # Simpan model
    full_model_path = f"models/{model_name}_full_model.keras"
    model.save(full_model_path)
    print(f"Model disimpan di: {full_model_path}")

    # Simpan vectorizer
    vectorizer_path = f"models/{model_name}_vectorizer.keras"
    tf.keras.models.save_model(
        tf.keras.Sequential([vectorizer], name="text_vectorization_model"),
        vectorizer_path,
    )
    print(f"Vectorizer disimpan di: {vectorizer_path}")

    return (
        model,
        history,
        test_pred_probs,
        test_labels,
        vectorizer,
        {"test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1},
    )


def plot_training_history(history, model_name):
    """Plot dan simpan kurva history training"""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Loss Training")
    plt.plot(history.history["val_loss"], label="Loss Validasi")
    plt.title(f"Kurva Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Akurasi Training")
    plt.plot(history.history["val_accuracy"], label="Akurasi Validasi")
    plt.title(f"Kurva Akurasi")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.legend()

    plt.tight_layout()
    plt.show()
