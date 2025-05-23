import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
import os

# Load modul lokal
from data_preprocessing import load_data, compute_f1_score


class SimpleRNNFromScratch:
    def __init__(self, keras_model_path, vectorizer_path):
        print(f"Loading Keras model from: {keras_model_path}")
        self.keras_model = tf.keras.models.load_model(keras_model_path)
        self.keras_model.summary()

        # Ekstrak bobot dari setiap layer model Keras
        self.extract_weights_from_keras_model()

        # Muat vectorizer untuk pemrosesan teks
        print(f"Loading vectorizer from: {vectorizer_path}")
        self.vectorizer_model = tf.keras.models.load_model(vectorizer_path)
        self.vectorizer = self.vectorizer_model.layers[0]

        print("Model loaded and weights extracted successfully")

    def extract_weights_from_keras_model(self):
        self.weights = {}

        # Ambil nama layer dari summary
        layer_names = [layer.name for layer in self.keras_model.layers]
        print(f"Layers in model: {layer_names}")

        # Cari layer embedding
        embedding_layer = self.keras_model.get_layer("embedding_layer")
        self.weights["embedding"] = embedding_layer.get_weights()[
            0
        ]  # Matriks embedding
        print(f"Embedding weights shape: {self.weights['embedding'].shape}")

        # Tangani multiple bidirectional layers
        self.bidirectional_layers = []
        for i, layer_name in enumerate(layer_names):
            if "bidirectional" in layer_name:
                self.bidirectional_layers.append(layer_name)

        print(f"Found {len(self.bidirectional_layers)} bidirectional layers")

        # Ekstrak bobot untuk semua layer bidirectional
        for i, layer_name in enumerate(self.bidirectional_layers):
            bidirectional_layer = self.keras_model.get_layer(layer_name)
            forward_layer = bidirectional_layer.forward_layer
            backward_layer = bidirectional_layer.backward_layer

            # Bobot forward
            prefix = f"layer{i+1}_"  # Tambahkan nomor layer ke nama bobot
            self.weights[prefix + "forward_rnn_kernel"] = forward_layer.get_weights()[0]
            self.weights[prefix + "forward_rnn_recurrent_kernel"] = (
                forward_layer.get_weights()[1]
            )
            self.weights[prefix + "forward_rnn_bias"] = forward_layer.get_weights()[2]

            # Bobot backward
            self.weights[prefix + "backward_rnn_kernel"] = backward_layer.get_weights()[
                0
            ]
            self.weights[prefix + "backward_rnn_recurrent_kernel"] = (
                backward_layer.get_weights()[1]
            )
            self.weights[prefix + "backward_rnn_bias"] = backward_layer.get_weights()[2]

            print(
                f"Layer {i+1} - Forward RNN kernel shape: {self.weights[prefix + 'forward_rnn_kernel'].shape}"
            )
            print(
                f"Layer {i+1} - Forward RNN recurrent kernel shape: {self.weights[prefix + 'forward_rnn_recurrent_kernel'].shape}"
            )

        # Ekstrak bobot layer dense output
        output_layer = self.keras_model.get_layer("output_dense_layer")
        self.weights["output_kernel"] = output_layer.get_weights()[0]
        self.weights["output_bias"] = output_layer.get_weights()[1]
        print(f"Output kernel shape: {self.weights['output_kernel'].shape}")
        print(f"Output bias shape: {self.weights['output_bias'].shape}")

    def embedding_forward(self, indices):
        # Lookup vektor embedding untuk setiap indeks
        # Setiap baris i di indices dipetakan ke vektor embedding yang sesuai
        return np.array([self.weights["embedding"][idx] for idx in indices])

    def simple_rnn_forward(self, inputs, kernel, recurrent_kernel, bias):
        batch_size, seq_length, input_dim = inputs.shape
        units = recurrent_kernel.shape[1]

        # Inisialisasi hidden state dengan nol
        h_t = np.zeros((batch_size, units))

        # Proses setiap time step
        for t in range(seq_length):
            # Ambil input pada time step saat ini
            x_t = inputs[:, t, :]

            # Hitung proyeksi input
            input_projection = np.dot(x_t, kernel)

            # Hitung proyeksi recurrent
            recurrent_projection = np.dot(h_t, recurrent_kernel)

            # SimpleRNN step dengan bias yang di-broadcast secara eksplisit
            h_t = np.tanh(
                input_projection + recurrent_projection + np.reshape(bias, (1, -1))
            )

        return h_t

    def bidirectional_rnn_forward(self, inputs, layer_idx=1):
        prefix = f"layer{layer_idx}_"

        # Forward pass
        forward_output = self.simple_rnn_forward(
            inputs,
            self.weights[prefix + "forward_rnn_kernel"],
            self.weights[prefix + "forward_rnn_recurrent_kernel"],
            self.weights[prefix + "forward_rnn_bias"],
        )

        # Backward pass (balik urutan input)
        reversed_inputs = inputs.copy()
        reversed_inputs = reversed_inputs[:, ::-1, :]
        backward_output = self.simple_rnn_forward(
            reversed_inputs,
            self.weights[prefix + "backward_rnn_kernel"],
            self.weights[prefix + "backward_rnn_recurrent_kernel"],
            self.weights[prefix + "backward_rnn_bias"],
        )

        # Gabungkan output forward dan backward
        return np.concatenate([forward_output, backward_output], axis=1)

    def dense_forward(self, inputs, kernel, bias):
        # Transformasi linear
        logits = np.dot(inputs, kernel) + bias

        # Aktivasi softmax dengan stabilitas numerik
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilities

    def forward(self, text_sequences):
        # Layer embedding
        embedded = self.embedding_forward(text_sequences)
        print(
            f"Embedded shape: {embedded.shape}, min: {embedded.min()}, max: {embedded.max()}, mean: {embedded.mean()}"
        )

        # Proses layer bidirectional pertama
        if len(self.bidirectional_layers) >= 1:
            # Layer pertama mengharapkan sequences
            # Untuk layer pertama, pertahankan dimensi sequence untuk layer berikutnya
            batch_size, seq_length, input_dim = embedded.shape
            units_per_direction = self.weights[
                "layer1_forward_rnn_recurrent_kernel"
            ].shape[0]

            # Proses setiap time step untuk layer bidirectional pertama
            # Untuk layer pertama kita perlu menyimpan output sequence
            forward_outputs = np.zeros((batch_size, seq_length, units_per_direction))
            backward_outputs = np.zeros((batch_size, seq_length, units_per_direction))

            # Arah forward
            h_t = np.zeros((batch_size, units_per_direction))
            for t in range(seq_length):
                x_t = embedded[:, t, :]
                h_t = np.tanh(
                    np.dot(x_t, self.weights["layer1_forward_rnn_kernel"])
                    + np.dot(h_t, self.weights["layer1_forward_rnn_recurrent_kernel"])
                    + self.weights["layer1_forward_rnn_bias"]
                )
                forward_outputs[:, t, :] = h_t

            # Arah backward (proses sequence secara terbalik)
            h_t = np.zeros((batch_size, units_per_direction))
            for t in range(seq_length - 1, -1, -1):
                x_t = embedded[:, t, :]
                h_t = np.tanh(
                    np.dot(x_t, self.weights["layer1_backward_rnn_kernel"])
                    + np.dot(h_t, self.weights["layer1_backward_rnn_recurrent_kernel"])
                    + self.weights["layer1_backward_rnn_bias"]
                )
                backward_outputs[:, t, :] = h_t

            # Gabungkan output forward dan backward sepanjang dimensi terakhir
            first_layer_output = np.concatenate(
                [forward_outputs, backward_outputs], axis=2
            )

            # Proses layer bidirectional kedua (jika ada)
            if len(self.bidirectional_layers) >= 2:
                # Layer kedua memproses seluruh sequence dan mengembalikan state akhir saja
                second_layer_output = self.bidirectional_rnn_forward(
                    first_layer_output, layer_idx=2
                )
                rnn_output = second_layer_output
            else:
                # Jika hanya satu layer, gunakan state akhir
                rnn_output = np.concatenate(
                    [
                        forward_outputs[:, -1, :],  # Time step terakhir dari forward
                        backward_outputs[
                            :, 0, :
                        ],  # Time step pertama dari backward (diproses terakhir)
                    ],
                    axis=1,
                )
        else:
            # Fallback untuk non-bidirectional (seharusnya tidak terjadi dengan model kita)
            rnn_output = self.simple_rnn_forward(
                embedded,
                self.weights["rnn_kernel"],
                self.weights["rnn_recurrent_kernel"],
                self.weights["rnn_bias"],
            )

        print(
            f"RNN output shape: {rnn_output.shape}, min: {rnn_output.min()}, max: {rnn_output.max()}, mean: {rnn_output.mean()}"
        )

        # Layer output dengan softmax
        logits = (
            np.dot(rnn_output, self.weights["output_kernel"])
            + self.weights["output_bias"]
        )
        print(f"Logits: {logits[:2]}")  # Cetak 2 contoh pertama

        # Terapkan softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Debug keberagaman prediksi
        unique_predictions = np.unique(np.argmax(output, axis=1))
        print(f"Unique predicted classes: {unique_predictions}")

        return output

    def predict(self, texts):
        # Konversi teks ke sequences menggunakan vectorizer
        sequences = self.vectorizer(texts).numpy()

        # Forward pass untuk mendapatkan probabilitas
        return self.forward(sequences)

    def compare_with_keras(self, texts):
        # Ambil prediksi dari implementasi from-scratch
        start_time_scratch = time.time()
        scratch_preds = self.predict(texts)
        scratch_time = time.time() - start_time_scratch

        # Ambil prediksi dari Keras
        start_time_keras = time.time()
        sequences = self.vectorizer(texts)
        keras_preds = self.keras_model.predict(sequences)
        keras_time = time.time() - start_time_keras

        # Hitung akurasi antara kedua implementasi
        scratch_classes = np.argmax(scratch_preds, axis=1)
        keras_classes = np.argmax(keras_preds, axis=1)
        accuracy = np.mean(scratch_classes == keras_classes)

        print(f"\nPerbandingan waktu prediksi:")
        print(f"From scratch: {scratch_time:.4f} detik")
        print(f"Keras: {keras_time:.4f} detik")
        print(f"Rasio waktu (scratch/keras): {scratch_time/keras_time:.2f}x")

        print(f"\nAkurasi kecocokan implementasi: {accuracy:.4f}")

        # Hitung mean absolute error antar prediksi
        mae = np.mean(np.abs(scratch_preds - keras_preds))
        print(f"Mean absolute error antar prediksi: {mae:.6f}")

        return scratch_preds, keras_preds, accuracy


def run_from_scratch_comparison(model_path, vectorizer_path):
    print("\n" + "=" * 50)
    print("SIMPLE RNN FROM SCRATCH IMPLEMENTATION")
    print("=" * 50)

    # Buat direktori output
    output_dir = "from_scratch_results"
    os.makedirs(output_dir, exist_ok=True)

    # Muat data
    (
        (train_texts, train_labels),
        (valid_texts, valid_labels),
        (test_texts, test_labels),
        label_mapping,
        num_classes,
    ) = load_data()
    print(f"Loaded test data: {len(test_texts)} samples")

    # Inisialisasi implementasi from-scratch
    scratch_rnn = SimpleRNNFromScratch(model_path, vectorizer_path)

    # Bandingkan implementasi pada data test
    print("\nMembandingkan implementasi pada data test...")
    scratch_preds, keras_preds, implementation_accuracy = (
        scratch_rnn.compare_with_keras(test_texts)
    )

    # Ambil prediksi kelas
    scratch_classes = np.argmax(scratch_preds, axis=1)
    keras_classes = np.argmax(keras_preds, axis=1)

    # Hitung metrik untuk kedua implementasi
    scratch_accuracy = np.mean(scratch_classes == test_labels)
    keras_accuracy = np.mean(keras_classes == test_labels)

    scratch_f1 = compute_f1_score(test_labels, scratch_preds)
    keras_f1 = compute_f1_score(test_labels, keras_preds)

    print("\nMetrik pada set test:")
    print(f"From scratch - Akurasi: {scratch_accuracy:.4f}, F1 Score: {scratch_f1:.4f}")
    print(f"Keras - Akurasi: {keras_accuracy:.4f}, F1 Score: {keras_f1:.4f}")

    # Tampilkan laporan klasifikasi
    class_names = list(label_mapping.keys())

    print("\nFrom scratch - Laporan Klasifikasi:")
    print(classification_report(test_labels, scratch_classes, target_names=class_names))

    print("\nKeras - Laporan Klasifikasi:")
    print(classification_report(test_labels, keras_classes, target_names=class_names))

    # Buat dan simpan confusion matrix
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm_scratch = confusion_matrix(test_labels, scratch_classes)
    plt.imshow(cm_scratch, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("From Scratch Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.subplot(1, 2, 2)
    cm_keras = confusion_matrix(test_labels, keras_classes)
    plt.imshow(cm_keras, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Keras Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices_comparison.png")

    # Bandingkan distribusi prediksi
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(np.max(scratch_preds, axis=1), bins=20, alpha=0.7)
    plt.title("From Scratch: Distribusi Probabilitas Maksimum")
    plt.xlabel("Probabilitas Maks")
    plt.ylabel("Jumlah")

    plt.subplot(1, 2, 2)
    plt.hist(np.max(keras_preds, axis=1), bins=20, alpha=0.7)
    plt.title("Keras: Distribusi Probabilitas Maksimum")
    plt.xlabel("Probabilitas Maks")
    plt.ylabel("Jumlah")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_distributions.png")

    # Bandingkan prediksi individual
    plt.figure(figsize=(10, 6))

    # Ambil beberapa sampel
    sample_size = min(20, len(test_texts))
    sample_indices = np.random.choice(len(test_texts), sample_size, replace=False)

    # Ambil probabilitas untuk setiap kelas dari scratch dan keras
    scratch_sample_preds = scratch_preds[sample_indices]
    keras_sample_preds = keras_preds[sample_indices]

    # Hitung perbedaan absolut antar implementasi
    abs_diffs = np.abs(scratch_sample_preds - keras_sample_preds)
    mean_diffs = np.mean(abs_diffs, axis=1)

    plt.bar(range(sample_size), mean_diffs)
    plt.title("Perbedaan Absolut Rata-rata dalam Prediksi")
    plt.xlabel("Indeks Sampel")
    plt.ylabel("Perbedaan Absolut Rata-rata")
    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_differences.png")

    # Simpan hasil detail ke file teks
    with open(f"{output_dir}/detailed_results.txt", "w") as f:
        f.write("SIMPLE RNN FROM SCRATCH EVALUATION\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Implementation match accuracy: {implementation_accuracy:.4f}\n")
        f.write(
            f"Mean absolute error between predictions: {np.mean(np.abs(scratch_preds - keras_preds)):.6f}\n\n"
        )

        f.write("Test set metrics:\n")
        f.write(
            f"From scratch - Accuracy: {scratch_accuracy:.4f}, F1 Score: {scratch_f1:.4f}\n"
        )
        f.write(f"Keras - Accuracy: {keras_accuracy:.4f}, F1 Score: {keras_f1:.4f}\n\n")

        f.write("From scratch - Classification Report:\n")
        f.write(
            classification_report(
                test_labels, scratch_classes, target_names=class_names
            )
        )
        f.write("\n")

        f.write("Keras - Classification Report:\n")
        f.write(
            classification_report(test_labels, keras_classes, target_names=class_names)
        )
        f.write("\n")

        f.write("From scratch - Confusion Matrix:\n")
        f.write(str(cm_scratch))
        f.write("\n\n")

        f.write("Keras - Confusion Matrix:\n")
        f.write(str(cm_keras))

    return scratch_rnn