import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_cifar10_data, compute_macro_f1_score
from model import create_cnn_model
import config
import os


def train_and_evaluate_cnn_model(
    input_shape=(224, 224, 3),
    num_conv_layers=config.NUM_CONV_LAYERS,
    base_filters=config.BASE_FILTERS,
    kernel_size=config.KERNEL_SIZE,
    pool_size=config.POOL_SIZE,
    dense_units=config.DENSE_UNITS,
    num_dense_layers=config.NUM_DENSE_LAYERS,
    dropout_rate=config.DROPOUT_RATE,
    l2_reg=config.L2_REG,
    learning_rate=config.LEARNING_RATE,
    model_name="cnn_classifier",
    use_class_weights=True,
    use_data_augmentation=True,
):
    """
    Fungsi untuk melatih dan mengevaluasi model CNN
    
    Parameters:
    -----------
    input_shape : tuple
        Bentuk input gambar (height, width, channels)
    num_conv_layers : int
        Jumlah blok konvolusi
    base_filters : int
        Jumlah filter awal
    kernel_size : tuple
        Ukuran kernel konvolusi
    pool_size : tuple
        Ukuran pooling window
    dense_units : int
        Jumlah unit di dense layer
    num_dense_layers : int
        Jumlah dense layer sebelum output
    dropout_rate : float
        Rate untuk dropout layer
    l2_reg : float
        Koefisien regularisasi L2
    learning_rate : float
        Learning rate untuk optimizer
    model_name : str
        Nama model untuk penyimpanan
    use_class_weights : bool
        Apakah menggunakan class weights untuk mengatasi imbalanced data
    use_data_augmentation : bool
        Apakah menggunakan data augmentation
    
    Returns:
    --------
    tuple
        (model, history, test_pred_probs, test_labels, metrics_dict)
    """
    
    print(f"\n--- Training CNN Model: {model_name} ---")
    print(
        f"Config: Conv Layers={num_conv_layers}, Base Filters={base_filters}, "
        f"Dense Units={dense_units}, Dense Layers={num_dense_layers}, "
        f"Dropout={dropout_rate}, L2={l2_reg}"
    )

    # Muat dataset gambar
    (
        (train_images, train_labels),
        (valid_images, valid_labels),
        (test_images, test_labels),
        label_mapping,
        num_classes,
    ) = load_cifar10_data(target_size=input_shape[:2])

    print(
        f"Ukuran dataset: Train={len(train_images)}, Valid={len(valid_images)}, Test={len(test_images)}"
    )
    print(f"Distribusi kelas di data train: {np.bincount(train_labels)}")
    print(f"Shape gambar: {train_images.shape}")

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

    # Normalisasi data gambar ke range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    valid_images = valid_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Data augmentation untuk training set
    if use_data_augmentation:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ], name="data_augmentation")
        print("Menggunakan data augmentation")
    else:
        data_augmentation = None

    # Buat dataset TF untuk efisiensi training
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    
    if use_data_augmentation:
        # Terapkan augmentation hanya pada training data
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    train_dataset = (
        train_dataset.shuffle(len(train_images) * 4, reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
    valid_dataset = valid_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Buat model CNN sesuai konfigurasi
    model = create_cnn_model(
        input_shape=input_shape,
        num_conv_layers=num_conv_layers,
        base_filters=base_filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dense_units=dense_units,
        num_dense_layers=num_dense_layers,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    # Tampilkan ringkasan model
    model.summary()

    # Buat direktori untuk checkpoint dan model jika belum ada
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

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
    print("Mulai pelatihan model CNN...")
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
    test_f1 = compute_macro_f1_score(test_labels, test_pred_probs)

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

    # Tampilkan confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Simpan model
    full_model_path = f"models/{model_name}_full_model.keras"
    model.save(full_model_path)
    print(f"Model disimpan di: {full_model_path}")

    return (
        model,
        history,
        test_pred_probs,
        test_labels,
        {"test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1},
    )


def plot_cnn_training_history(history, model_name):
    """Plot dan simpan kurva history training untuk CNN"""
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history["loss"], label="Loss Training", color='blue')
    plt.plot(history.history["val_loss"], label="Loss Validasi", color='red')
    plt.title(f"Kurva Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history["accuracy"], label="Akurasi Training", color='blue')
    plt.plot(history.history["val_accuracy"], label="Akurasi Validasi", color='red')
    plt.title(f"Kurva Akurasi - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Learning Rate (jika ada)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history["lr"], label="Learning Rate", color='green')
        plt.title(f"Learning Rate - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, "Learning Rate\ndata not available", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Learning Rate")

    plt.tight_layout()
    
    # Simpan plot
    plot_path = f"plots/{model_name}_training_history.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot training history disimpan di: {plot_path}")
    
    plt.show()


def visualize_predictions(model, test_images, test_labels, label_mapping, num_samples=16):
    """
    Visualisasi prediksi model pada beberapa sample test data
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Model yang sudah dilatih
    test_images : numpy.ndarray
        Data gambar test
    test_labels : numpy.ndarray
        Label sebenarnya
    label_mapping : dict
        Mapping dari index ke nama kelas
    num_samples : int
        Jumlah sample yang akan ditampilkan
    """
    # Ambil sample random
    indices = np.random.choice(len(test_images), size=num_samples, replace=False)
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]
    
    # Prediksi
    predictions = model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Buat mapping dari index ke nama kelas
    idx_to_class = {v: k for k, v in label_mapping.items()}
    
    # Plot hasil
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Denormalisasi gambar untuk ditampilkan
        img = sample_images[i]
        if img.max() <= 1.0:  # Jika sudah dinormalisasi
            img = (img * 255).astype(np.uint8)
            
        axes[i].imshow(img)
        
        true_label = idx_to_class[sample_labels[i]]
        pred_label = idx_to_class[predicted_labels[i]]
        confidence = predictions[i][predicted_labels[i]] * 100
        
        # Warna hijau jika benar, merah jika salah
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# Contoh penggunaan
def main():
    """Contoh penggunaan fungsi train_and_evaluate_cnn_model"""
    
    # Training dengan konfigurasi default
    model, history, test_pred_probs, test_labels, metrics = train_and_evaluate_cnn_model(
        model_name="cnn_baseline",
        use_class_weights=True,
        use_data_augmentation=True
    )
    
    # Plot training history
    plot_cnn_training_history(history, "cnn_baseline")
    
    # Visualisasi prediksi (asumsi test_images tersedia)
    # visualize_predictions(model, test_images, test_labels, label_mapping)
    
    print("Training selesai!")
    return model, history, metrics


if __name__ == "__main__":
    main()