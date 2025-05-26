import numpy as np
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Impor dari modul lokal di direktori yang sama
import config  # Menggunakan config.py dari folder lstm/
from data_preprocessing import (
    load_data,
    compute_f1_score,
)  # compute_f1_score dari data_preprocessing.py


# Fungsi aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class LSTMFromScratch:
    def __init__(self, keras_model_path, vectorizer_path):
        print(f"Loading Keras LSTM model from: {keras_model_path}")
        self.keras_model = tf.keras.models.load_model(keras_model_path, compile=False)

        print(f"Loading vectorizer from: {vectorizer_path}")
        self.vectorizer_model = tf.keras.models.load_model(
            vectorizer_path, compile=False
        )
        self.vectorizer = self.vectorizer_model.layers[0]

        self.weights = {}
        self.lstm_layer_configs = []
        self.extract_weights_from_keras_model()
        print("LSTM Model loaded and weights extracted successfully.")

    def extract_weights_from_keras_model(self):
        print("\nExtracting weights from Keras LSTM model...")
        lstm_stack_idx = 0

        for layer in self.keras_model.layers:
            layer_name = layer.name
            print(f"Processing layer: {layer_name} (Type: {type(layer).__name__})")

            if isinstance(layer, tf.keras.layers.Embedding):
                if layer.name == "embedding_layer":  # Sesuai model_lstm.py
                    self.weights["embedding"] = layer.get_weights()[0]
                    print(
                        f"  Extracted embedding weights, shape: {self.weights['embedding'].shape}"
                    )

            elif isinstance(layer, tf.keras.layers.Bidirectional) and isinstance(
                layer.forward_layer, tf.keras.layers.LSTM
            ):
                lstm_stack_idx += 1
                fwd_lstm_layer = layer.forward_layer
                bwd_lstm_layer = layer.backward_layer

                config_entry = {
                    "type": "bidirectional",
                    "keras_layer_name": layer_name,
                    "stack_idx": lstm_stack_idx,
                    "fwd_prefix": f"bidir_fwd_lstm_{lstm_stack_idx}_",
                    "bwd_prefix": f"bidir_bwd_lstm_{lstm_stack_idx}_",
                    "return_sequences": layer.return_sequences,
                }
                self.lstm_layer_configs.append(config_entry)

                # Bobot Forward LSTM
                f_kernel, f_recurrent_kernel, f_bias = fwd_lstm_layer.get_weights()
                self.weights[config_entry["fwd_prefix"] + "kernel"] = f_kernel
                self.weights[config_entry["fwd_prefix"] + "recurrent_kernel"] = (
                    f_recurrent_kernel
                )
                self.weights[config_entry["fwd_prefix"] + "bias"] = f_bias
                print(
                    f"  Extracted Bidirectional Forward LSTM ({fwd_lstm_layer.name}) weights with prefix '{config_entry['fwd_prefix']}'"
                )

                # Bobot Backward LSTM
                b_kernel, b_recurrent_kernel, b_bias = bwd_lstm_layer.get_weights()
                self.weights[config_entry["bwd_prefix"] + "kernel"] = b_kernel
                self.weights[config_entry["bwd_prefix"] + "recurrent_kernel"] = (
                    b_recurrent_kernel
                )
                self.weights[config_entry["bwd_prefix"] + "bias"] = b_bias
                print(
                    f"  Extracted Bidirectional Backward LSTM ({bwd_lstm_layer.name}) weights with prefix '{config_entry['bwd_prefix']}'"
                )

            elif isinstance(layer, tf.keras.layers.LSTM) and not any(
                layer.name
                in (
                    self.keras_model.get_layer(
                        cfg["keras_layer_name"]
                    ).forward_layer.name,
                    self.keras_model.get_layer(
                        cfg["keras_layer_name"]
                    ).backward_layer.name,
                )
                for cfg in self.lstm_layer_configs
                if cfg["type"] == "bidirectional"
            ):

                lstm_stack_idx += 1
                config_entry = {
                    "type": "unidirectional",
                    "keras_layer_name": layer_name,
                    "stack_idx": lstm_stack_idx,
                    "prefix": f"uni_lstm_{lstm_stack_idx}_",
                    "return_sequences": layer.return_sequences,
                }
                self.lstm_layer_configs.append(config_entry)

                kernel, recurrent_kernel, bias = layer.get_weights()
                self.weights[config_entry["prefix"] + "kernel"] = kernel
                self.weights[config_entry["prefix"] + "recurrent_kernel"] = (
                    recurrent_kernel
                )
                self.weights[config_entry["prefix"] + "bias"] = bias
                print(
                    f"  Extracted Unidirectional LSTM ({layer_name}) weights with prefix '{config_entry['prefix']}'"
                )

            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.name == "output_dense_layer":  # Sesuai model_lstm.py
                    self.weights["output_kernel"] = layer.get_weights()[0]
                    self.weights["output_bias"] = layer.get_weights()[1]
                    print(
                        f"  Extracted output dense layer weights. Kernel: {self.weights['output_kernel'].shape}, Bias: {self.weights['output_bias'].shape}"
                    )

        # Urutkan konfigurasi layer LSTM berdasarkan stack_idx untuk memastikan urutan pemrosesan yang benar
        self.lstm_layer_configs.sort(key=lambda x: x["stack_idx"])
        print(
            f"Finished weight extraction. Found {len(self.lstm_layer_configs)} LSTM stack configurations."
        )

    def embedding_forward(self, indices):
        return self.weights["embedding"][indices]

    def _lstm_cell_forward(self, x_t, h_prev, c_prev, W_all, U_all, b_all):
        """
        Satu langkah forward untuk LSTM cell.
        W_all: Gabungan kernel input (input_dim, 4*units)
        U_all: Gabungan kernel recurrent (units, 4*units)
        b_all: Gabungan bias (4*units,)
        Urutan gate di Keras: i, f, c, o
        """
        units = U_all.shape[0]

        # Proyeksi gabungan
        z = np.dot(x_t, W_all) + np.dot(h_prev, U_all) + b_all

        # Input gate
        i = sigmoid(z[:, :units])
        # Forget gate
        f = sigmoid(z[:, units : 2 * units])
        # Candidate cell state (g atau c_tilde)
        c_candidate = tanh(z[:, 2 * units : 3 * units])
        # Output gate
        o = sigmoid(z[:, 3 * units :])

        # Update cell state
        c_t = f * c_prev + i * c_candidate
        # Update hidden state
        h_t = o * tanh(c_t)

        return h_t, c_t

    def _lstm_pass(
        self, inputs, W_all, U_all, b_all, return_sequences=True, go_backwards=False
    ):
        batch_size, seq_length, _ = inputs.shape
        units = U_all.shape[0]  # Hidden units

        h_t = np.zeros((batch_size, units))
        c_t = np.zeros((batch_size, units))

        if return_sequences:
            outputs_h = np.zeros((batch_size, seq_length, units))
            # outputs_c = np.zeros((batch_size, seq_length, units)) # Jika perlu cell states

        time_steps = range(seq_length)
        if go_backwards:
            time_steps = range(seq_length - 1, -1, -1)

        for t_idx, t_val in enumerate(time_steps):
            x_step = inputs[:, t_val, :]
            h_t, c_t = self._lstm_cell_forward(x_step, h_t, c_t, W_all, U_all, b_all)

            if return_sequences:
                # Simpan dalam urutan alami meskipun diproses terbalik
                storage_idx = t_val
                outputs_h[:, storage_idx, :] = h_t
                # outputs_c[:, storage_idx, :] = c_t

        return (
            outputs_h if return_sequences else h_t
        )  # Hanya mengembalikan hidden states

    def dense_forward(self, inputs, kernel, bias):
        logits = np.dot(inputs, kernel) + bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities

    def forward(self, text_sequences):
        # 1. Embedding Layer
        current_tensor = self.embedding_forward(text_sequences)
        # print(f"After Embedding: shape={current_tensor.shape}")

        # 2. LSTM Layers
        for config_entry in self.lstm_layer_configs:
            # print(f"Processing LSTM stack {config_entry['stack_idx']}: Type={config_entry['type']}, KerasReturnSeq={config_entry['return_sequences']}")

            # return_sequences untuk pass LSTM ini ditentukan oleh konfigurasi Keras layer
            return_sequences_for_this_pass = config_entry["return_sequences"]

            if config_entry["type"] == "bidirectional":
                fwd_W = self.weights[config_entry["fwd_prefix"] + "kernel"]
                fwd_U = self.weights[config_entry["fwd_prefix"] + "recurrent_kernel"]
                fwd_b = self.weights[config_entry["fwd_prefix"] + "bias"]

                bwd_W = self.weights[config_entry["bwd_prefix"] + "kernel"]
                bwd_U = self.weights[config_entry["bwd_prefix"] + "recurrent_kernel"]
                bwd_b = self.weights[config_entry["bwd_prefix"] + "bias"]

                # Forward pass LSTM
                h_fwd = self._lstm_pass(
                    current_tensor,
                    fwd_W,
                    fwd_U,
                    fwd_b,
                    return_sequences=True,
                    go_backwards=False,
                )  # Selalu True untuk digabungkan

                # Backward pass LSTM
                h_bwd = self._lstm_pass(
                    current_tensor,
                    bwd_W,
                    bwd_U,
                    bwd_b,
                    return_sequences=True,
                    go_backwards=True,
                )  # Selalu True untuk digabungkan

                # Gabungkan output
                if (
                    return_sequences_for_this_pass
                ):  # Jika Keras layer ini return_sequences=True
                    current_tensor = np.concatenate([h_fwd, h_bwd], axis=-1)
                else:  # Jika Keras layer ini return_sequences=False (biasanya layer LSTM terakhir)
                    # Ambil state terakhir dari forward pass dan state "pertama" (terakhir diproses) dari backward pass
                    current_tensor = np.concatenate(
                        [h_fwd[:, -1, :], h_bwd[:, 0, :]], axis=-1
                    )

            elif config_entry["type"] == "unidirectional":
                W = self.weights[config_entry["prefix"] + "kernel"]
                U = self.weights[config_entry["prefix"] + "recurrent_kernel"]
                b = self.weights[config_entry["prefix"] + "bias"]

                current_tensor = self._lstm_pass(
                    current_tensor,
                    W,
                    U,
                    b,
                    return_sequences=return_sequences_for_this_pass,
                    go_backwards=False,
                )
            # print(f"  After LSTM stack {config_entry['stack_idx']}: shape={current_tensor.shape}")

        lstm_output = current_tensor

        # 3. Output Dense Layer
        output = self.dense_forward(
            lstm_output, self.weights["output_kernel"], self.weights["output_bias"]
        )
        return output

    def predict(self, texts):
        sequences = self.vectorizer(texts).numpy()
        return self.forward(sequences)

    def compare_with_keras(self, texts):
        start_time_scratch = time.time()
        scratch_preds = self.predict(texts)
        scratch_time = time.time() - start_time_scratch

        start_time_keras = time.time()
        sequences_tf = self.vectorizer(texts)
        keras_preds = self.keras_model.predict(sequences_tf)
        keras_time = time.time() - start_time_keras

        scratch_classes = np.argmax(scratch_preds, axis=1)
        keras_classes = np.argmax(keras_preds, axis=1)

        implementation_accuracy = 0.0
        mae = float("inf")
        if not (np.any(np.isnan(scratch_preds)) or np.any(np.isinf(scratch_preds))):
            implementation_accuracy = np.mean(scratch_classes == keras_classes)
            mae = np.mean(np.abs(scratch_preds - keras_preds))
        else:
            print("WARNING: NaN or Inf found in scratch predictions!")

        print(f"\nPerbandingan waktu prediksi (LSTM):")
        print(f"From scratch: {scratch_time:.4f} detik")
        print(f"Keras: {keras_time:.4f} detik")
        if keras_time > 0:
            print(f"Rasio waktu (scratch/keras): {scratch_time/keras_time:.2f}x")
        else:
            print(f"Rasio waktu (scratch/keras): N/A (Keras time is zero)")

        print(
            f"\nAkurasi kecocokan implementasi (kelas): {implementation_accuracy:.6f}"
        )
        print(f"Mean absolute error antar probabilitas prediksi: {mae:.8f}")

        if (
            mae > 1e-4
        ):  # Threshold bisa disesuaikan, LSTM mungkin punya error numerik sedikit lebih besar
            print("  MAE tinggi, memeriksa beberapa perbedaan prediksi:")
            for i in range(min(3, scratch_preds.shape[0])):  # Tampilkan beberapa contoh
                print(
                    f"    Sample {i}: Scratch={scratch_preds[i]}, Keras={keras_preds[i]}, Diff MAE={np.mean(np.abs(scratch_preds[i] - keras_preds[i]))}"
                )

        return scratch_preds, keras_preds, implementation_accuracy, mae


def run_lstm_from_scratch_comparison(model_path, vectorizer_path):
    """
    Fungsi utama untuk menjalankan perbandingan implementasi LSTM from scratch dengan Keras.
    """
    print("\n" + "=" * 50)
    print("LSTM FROM SCRATCH IMPLEMENTATION COMPARISON (ROBUST)")
    print("=" * 50)

    output_dir = "from_scratch_lstm_results"  # Nama direktori output spesifik LSTM
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output akan disimpan di direktori: {output_dir}")

    (
        (train_texts, train_labels),
        (valid_texts, valid_labels),
        (test_texts, test_labels),
        label_mapping,
        num_classes,
    ) = load_data()  # Path data ditangani oleh load_data
    print(f"Loaded test data: {len(test_texts)} samples")

    scratch_lstm = LSTMFromScratch(model_path, vectorizer_path)

    print("\nMembandingkan implementasi LSTM pada data test...")

    scratch_preds, keras_preds, implementation_accuracy, mae = (
        scratch_lstm.compare_with_keras(test_texts)
    )

    scratch_classes = None
    cm_scratch = None  # Inisialisasi cm_scratch
    if not (np.any(np.isnan(scratch_preds)) or np.any(np.isinf(scratch_preds))):
        scratch_classes = np.argmax(scratch_preds, axis=1)
        cm_scratch = confusion_matrix(test_labels, scratch_classes)  # Hitung jika valid

    keras_classes = np.argmax(keras_preds, axis=1)
    cm_keras = confusion_matrix(
        test_labels, keras_classes
    )  # Selalu bisa dihitung untuk Keras

    scratch_accuracy = 0.0
    scratch_f1 = 0.0
    if scratch_classes is not None:
        scratch_accuracy = np.mean(scratch_classes == test_labels)
        scratch_f1 = compute_f1_score(
            test_labels, scratch_preds
        )  # compute_f1_score menerima probabilitas
    else:
        print(
            "Peringatan: Metrik Scratch LSTM tidak dapat dihitung karena prediksi tidak valid."
        )

    keras_accuracy = np.mean(keras_classes == test_labels)
    keras_f1 = compute_f1_score(test_labels, keras_preds)

    print("\nMetrik pada set test (LSTM):")
    print(
        f"From scratch - Akurasi: {scratch_accuracy:.4f}, F1 Score (Macro): {scratch_f1:.4f}"
    )
    print(f"Keras - Akurasi: {keras_accuracy:.4f}, F1 Score (Macro): {keras_f1:.4f}")

    class_names = list(label_mapping.keys())

    if scratch_classes is not None:
        print("\nFrom scratch LSTM - Laporan Klasifikasi:")
        print(
            classification_report(
                test_labels, scratch_classes, target_names=class_names, zero_division=0
            )
        )
    else:
        print(
            "\nFrom scratch LSTM - Laporan Klasifikasi: Tidak tersedia karena prediksi tidak valid."
        )

    print("\nKeras LSTM - Laporan Klasifikasi:")
    print(
        classification_report(
            test_labels, keras_classes, target_names=class_names, zero_division=0
        )
    )

    # Visualisasi hanya jika prediksi scratch valid
    if scratch_classes is not None and cm_scratch is not None:
        # Buat dan simpan confusion matrix
        plt.figure(figsize=(13, 5.5))

        plt.subplot(1, 2, 1)
        plt.imshow(cm_scratch, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("From Scratch LSTM Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        for i in range(cm_scratch.shape[0]):
            for j in range(cm_scratch.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm_scratch[i, j], "d"),
                    horizontalalignment="center",
                    color=(
                        "white"
                        if cm_scratch[i, j] > cm_scratch.max() / 2.0
                        else "black"
                    ),
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.imshow(cm_keras, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Keras LSTM Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        for i in range(cm_keras.shape[0]):
            for j in range(cm_keras.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm_keras[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm_keras[i, j] > cm_keras.max() / 2.0 else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        plt.suptitle("Perbandingan Confusion Matrix (LSTM)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{output_dir}/confusion_matrices_lstm_comparison.png")
        print(
            f"Plot confusion matrix LSTM disimpan di: {output_dir}/confusion_matrices_lstm_comparison.png"
        )
        plt.close()

        # Bandingkan distribusi prediksi
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(
            np.max(scratch_preds, axis=1), bins=20, alpha=0.7, label="Scratch LSTM"
        )
        plt.title("Scratch LSTM: Distribusi Probabilitas Maks")
        plt.xlabel("Probabilitas Maks")
        plt.ylabel("Jumlah")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(
            np.max(keras_preds, axis=1),
            bins=20,
            alpha=0.7,
            label="Keras LSTM",
            color="orange",
        )
        plt.title("Keras LSTM: Distribusi Probabilitas Maks")
        plt.xlabel("Probabilitas Maks")
        plt.ylabel("Jumlah")
        plt.legend()

        plt.suptitle(
            "Perbandingan Distribusi Probabilitas Maksimum (LSTM)", fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{output_dir}/probability_distributions_lstm.png")
        print(
            f"Plot distribusi probabilitas LSTM disimpan di: {output_dir}/probability_distributions_lstm.png"
        )
        plt.close()

        # Bandingkan prediksi individual
        plt.figure(figsize=(10, 6))
        sample_size = min(20, len(test_texts))
        if len(test_texts) > 0:
            sample_indices = np.random.choice(
                len(test_texts), sample_size, replace=False
            )
            scratch_sample_preds = scratch_preds[sample_indices]
            keras_sample_preds = keras_preds[sample_indices]
            abs_diffs = np.abs(scratch_sample_preds - keras_sample_preds)
            mean_diffs_per_sample = np.mean(abs_diffs, axis=1)

            plt.bar(range(sample_size), mean_diffs_per_sample)
            plt.title("Perbedaan Absolut Rata-rata Probabilitas per Sampel (LSTM)")
            plt.xlabel("Indeks Sampel Acak")
            plt.ylabel("MAE Probabilitas")
            plt.xticks(
                range(sample_size), sample_indices.astype(str), rotation=45, ha="right"
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/prediction_differences_lstm.png")
            print(
                f"Plot perbedaan prediksi LSTM disimpan di: {output_dir}/prediction_differences_lstm.png"
            )
        else:
            print(
                "Tidak ada data tes untuk memplot perbedaan prediksi individual (LSTM)."
            )
        plt.close()
    else:
        print("Plotting LSTM dilewati karena prediksi scratch tidak valid.")

    # Simpan hasil detail ke file teks
    results_filepath = f"{output_dir}/detailed_lstm_results.txt"
    with open(results_filepath, "w") as f:
        f.write("LSTM FROM SCRATCH EVALUATION (ROBUST)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Keras Model Path: {model_path}\n")
        f.write(f"Vectorizer Path: {vectorizer_path}\n\n")

        f.write(
            f"Implementation match accuracy (kelas): {implementation_accuracy:.6f}\n"
        )
        f.write(
            f"Mean absolute error (MAE) between prediction probabilities: {mae:.8f}\n\n"
        )

        f.write("Test set metrics (LSTM):\n")
        f.write(
            f"  From scratch - Accuracy: {scratch_accuracy:.4f}, F1 Score (Macro): {scratch_f1:.4f}\n"
        )
        f.write(
            f"  Keras        - Accuracy: {keras_accuracy:.4f}, F1 Score (Macro): {keras_f1:.4f}\n\n"
        )

        if scratch_classes is not None:
            f.write("From scratch LSTM - Classification Report:\n")
            f.write(
                classification_report(
                    test_labels,
                    scratch_classes,
                    target_names=class_names,
                    zero_division=0,
                )
            )
            f.write("\n\n")
        else:
            f.write(
                "From scratch LSTM - Classification Report: Not available due to invalid predictions.\n\n"
            )

        f.write("Keras LSTM - Classification Report:\n")
        f.write(
            classification_report(
                test_labels, keras_classes, target_names=class_names, zero_division=0
            )
        )
        f.write("\n\n")

        if cm_scratch is not None:
            cm_scratch_str = np.array2string(cm_scratch, separator=", ")
            f.write("From scratch LSTM - Confusion Matrix:\n")
            f.write(cm_scratch_str)
            f.write("\n\n")
        else:
            f.write(
                "From scratch LSTM - Confusion Matrix: Not available due to invalid predictions.\n\n"
            )

        cm_keras_str = np.array2string(cm_keras, separator=", ")
        f.write("Keras LSTM - Confusion Matrix:\n")
        f.write(cm_keras_str)
        f.write("\n")

    print(f"Hasil detail LSTM disimpan di: {results_filepath}")
    return scratch_lstm
