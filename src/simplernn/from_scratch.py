import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score as sklearn_f1_score,
)
import time
import os

# Load modul lokal
from data_preprocessing import (
    load_data,
    compute_f1_score,
)


class SimpleRNNFromScratch:
    def __init__(self, keras_model_path, vectorizer_path):
        print(f"Loading Keras model from: {keras_model_path}")
        self.keras_model = tf.keras.models.load_model(keras_model_path, compile=False)

        # Muat vectorizer untuk pemrosesan teks
        print(f"Loading vectorizer from: {vectorizer_path}")
        self.vectorizer_model = tf.keras.models.load_model(
            vectorizer_path, compile=False
        )
        self.vectorizer = self.vectorizer_model.layers[0]

        # Ekstrak bobot dari setiap layer model Keras
        self.weights = {}
        self.rnn_layer_configs = []
        self.extract_weights_from_keras_model()

        print("Model loaded and weights extracted successfully.")

    def extract_weights_from_keras_model(self):
        print("\nExtracting weights from Keras model...")
        rnn_layer_counter = 0
        for layer in self.keras_model.layers:
            layer_name = layer.name
            print(f"Processing layer: {layer_name} (Type: {type(layer).__name__})")

            if isinstance(layer, tf.keras.layers.Embedding):
                if layer_name == "embedding_layer":
                    self.weights["embedding"] = layer.get_weights()[0]
                    print(
                        f"  Extracted embedding weights, shape: {self.weights['embedding'].shape}"
                    )

            elif isinstance(layer, tf.keras.layers.Bidirectional):
                if isinstance(layer.forward_layer, tf.keras.layers.SimpleRNN):
                    rnn_layer_counter += 1
                    config = {
                        "type": "bidirectional",
                        "name": layer_name,
                        "keras_layer_name": layer_name,  # Nama layer Bidirectional di Keras
                        "weights_prefix": f"bidir_rnn_{rnn_layer_counter}_",
                    }

                    forward_rnn = layer.forward_layer
                    backward_rnn = layer.backward_layer

                    prefix = config["weights_prefix"]
                    self.weights[prefix + "fwd_kernel"] = forward_rnn.get_weights()[0]
                    self.weights[prefix + "fwd_recurrent_kernel"] = (
                        forward_rnn.get_weights()[1]
                    )
                    self.weights[prefix + "fwd_bias"] = forward_rnn.get_weights()[2]

                    self.weights[prefix + "bwd_kernel"] = backward_rnn.get_weights()[0]
                    self.weights[prefix + "bwd_recurrent_kernel"] = (
                        backward_rnn.get_weights()[1]
                    )
                    self.weights[prefix + "bwd_bias"] = backward_rnn.get_weights()[2]

                    self.rnn_layer_configs.append(config)
                    print(
                        f"  Extracted Bidirectional SimpleRNN weights for '{layer_name}' with prefix '{prefix}'"
                    )
                    print(
                        f"    Forward kernel shape: {self.weights[prefix + 'fwd_kernel'].shape}"
                    )
                    print(
                        f"    Backward kernel shape: {self.weights[prefix + 'bwd_kernel'].shape}"
                    )

            elif isinstance(layer, tf.keras.layers.SimpleRNN) and not any(
                layer_name in cfg["keras_layer_name"]
                for cfg in self.rnn_layer_configs
                if "keras_layer_name" in cfg and layer_name in cfg["keras_layer_name"]
            ):
                is_part_of_bidirectional = False
                for keras_layer_outer in self.keras_model.layers:
                    if isinstance(
                        keras_layer_outer, tf.keras.layers.Bidirectional
                    ) and (
                        layer == keras_layer_outer.forward_layer
                        or layer == keras_layer_outer.backward_layer
                    ):
                        is_part_of_bidirectional = True
                        break
                if is_part_of_bidirectional:
                    continue

                rnn_layer_counter += 1
                config = {
                    "type": "unidirectional",
                    "name": layer_name,  # Nama layer SimpleRNN di Keras
                    "keras_layer_name": layer_name,
                    "weights_prefix": f"unidir_rnn_{rnn_layer_counter}_",
                }

                prefix = config["weights_prefix"]
                self.weights[prefix + "kernel"] = layer.get_weights()[0]
                self.weights[prefix + "recurrent_kernel"] = layer.get_weights()[1]
                self.weights[prefix + "bias"] = layer.get_weights()[2]

                self.rnn_layer_configs.append(config)
                print(
                    f"  Extracted Unidirectional SimpleRNN weights for '{layer_name}' with prefix '{prefix}'"
                )
                print(f"    Kernel shape: {self.weights[prefix + 'kernel'].shape}")

            elif isinstance(layer, tf.keras.layers.Dense):
                if layer_name == "output_dense_layer":  
                    self.weights["output_kernel"] = layer.get_weights()[0]
                    self.weights["output_bias"] = layer.get_weights()[1]
                    print(
                        f"  Extracted output dense layer weights. Kernel: {self.weights['output_kernel'].shape}, Bias: {self.weights['output_bias'].shape}"
                    )

        print(
            f"Finished weight extraction. Found {len(self.rnn_layer_configs)} RNN layer configurations."
        )

    def embedding_forward(self, indices):
        return self.weights["embedding"][indices]

    def _simple_rnn_pass(
        self,
        inputs,
        kernel,
        recurrent_kernel,
        bias,
        return_sequences=True,
        go_backwards=False,
    ):
        batch_size, seq_length, _ = inputs.shape
        units = bias.shape[0]

        h_t = np.zeros((batch_size, units))

        if return_sequences:
            outputs = np.zeros((batch_size, seq_length, units))

        time_steps = range(seq_length)
        if go_backwards:
            time_steps = range(seq_length - 1, -1, -1)

        for t in time_steps:
            x_t = inputs[:, t, :]
            h_t = np.tanh(np.dot(x_t, kernel) + np.dot(h_t, recurrent_kernel) + bias)
            if return_sequences:
                if go_backwards:
                    outputs[:, t, :] = h_t
                else:
                    outputs[:, t, :] = h_t

        return outputs if return_sequences else h_t

    def dense_forward(self, inputs, kernel, bias):
        logits = np.dot(inputs, kernel) + bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities

    def forward(self, text_sequences):
        # 1. Embedding Layer
        current_tensor = self.embedding_forward(text_sequences)

        # 2. RNN Layers (Bidirectional or Unidirectional)
        num_rnn_layers = len(self.rnn_layer_configs)
        for i, rnn_config in enumerate(self.rnn_layer_configs):
            is_last_rnn_in_stack = i == num_rnn_layers - 1
            return_sequences_for_this_layer = not is_last_rnn_in_stack

            prefix = rnn_config["weights_prefix"]

            if rnn_config["type"] == "bidirectional":
                fwd_kernel = self.weights[prefix + "fwd_kernel"]
                fwd_recurrent_kernel = self.weights[prefix + "fwd_recurrent_kernel"]
                fwd_bias = self.weights[prefix + "fwd_bias"]

                bwd_kernel = self.weights[prefix + "bwd_kernel"]
                bwd_recurrent_kernel = self.weights[prefix + "bwd_recurrent_kernel"]
                bwd_bias = self.weights[prefix + "bwd_bias"]

                # Forward pass
                h_fwd = self._simple_rnn_pass(
                    current_tensor,
                    fwd_kernel,
                    fwd_recurrent_kernel,
                    fwd_bias,
                    return_sequences=True,
                    go_backwards=False,
                )

                # Backward pass
                h_bwd = self._simple_rnn_pass(
                    current_tensor,
                    bwd_kernel,
                    bwd_recurrent_kernel,
                    bwd_bias,
                    return_sequences=True,
                    go_backwards=True,
                )

                if return_sequences_for_this_layer:
                    current_tensor = np.concatenate([h_fwd, h_bwd], axis=-1)
                else:  
                    current_tensor = np.concatenate(
                        [h_fwd[:, -1, :], h_bwd[:, 0, :]], axis=-1
                    )

            elif rnn_config["type"] == "unidirectional":
                kernel = self.weights[prefix + "kernel"]
                recurrent_kernel = self.weights[prefix + "recurrent_kernel"]
                bias = self.weights[prefix + "bias"]

                current_tensor = self._simple_rnn_pass(
                    current_tensor,
                    kernel,
                    recurrent_kernel,
                    bias,
                    return_sequences=return_sequences_for_this_layer,
                    go_backwards=False,
                )

        rnn_output = current_tensor

        # 3. Output Dense Layer
        output = self.dense_forward(
            rnn_output, self.weights["output_kernel"], self.weights["output_bias"]
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

        # Periksa apakah ada NaN atau Inf dalam prediksi scratch
        if np.any(np.isnan(scratch_preds)) or np.any(np.isinf(scratch_preds)):
            print("WARNING: NaN or Inf found in scratch predictions!")
            implementation_accuracy = 0.0
            mae = float("inf")
        else:
            implementation_accuracy = np.mean(scratch_classes == keras_classes)
            mae = np.mean(np.abs(scratch_preds - keras_preds))

        print(f"\nPerbandingan waktu prediksi:")
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

        # Debugging perbedaan jika MAE tinggi
        if mae > 1e-5:
            print("  MAE tinggi, memeriksa beberapa perbedaan prediksi:")
            for i in range(min(5, scratch_preds.shape[0])):
                print(
                    f"    Sample {i}: Scratch={scratch_preds[i]}, Keras={keras_preds[i]}, Diff={np.abs(scratch_preds[i] - keras_preds[i])}"
                )

        return scratch_preds, keras_preds, implementation_accuracy, mae


def run_from_scratch_comparison(model_path, vectorizer_path):
    print("\n" + "=" * 50)
    print(
        "SIMPLE RNN FROM SCRATCH IMPLEMENTATION"
    )  
    print("=" * 50)

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

    scratch_preds, keras_preds, implementation_accuracy, mae = (
        scratch_rnn.compare_with_keras(test_texts)
    )

    # Ambil prediksi kelas
    scratch_classes = None
    if scratch_preds is not None and not (
        np.any(np.isnan(scratch_preds)) or np.any(np.isinf(scratch_preds))
    ):
        scratch_classes = np.argmax(scratch_preds, axis=1)

    keras_classes = np.argmax(keras_preds, axis=1)

    # Hitung metrik untuk kedua implementasi
    scratch_accuracy = 0.0
    scratch_f1 = 0.0
    if scratch_classes is not None:
        scratch_accuracy = np.mean(scratch_classes == test_labels)
        scratch_f1 = compute_f1_score(  
            test_labels, scratch_preds
        )
    else:
        print(
            "Peringatan: Metrik Scratch tidak dapat dihitung karena prediksi tidak valid (NaN/Inf)."
        )

    keras_accuracy = np.mean(keras_classes == test_labels)
    keras_f1 = compute_f1_score(test_labels, keras_preds)

    # --- Mulai Laporan Detail ---
    report_lines = []
    report_lines.append("SIMPLE RNN FROM SCRATCH EVALUATION")
    report_lines.append("=" * 50 + "\n")
    report_lines.append(f"Keras Model Path: {model_path}")
    report_lines.append(f"Vectorizer Path: {vectorizer_path}\n")

    report_lines.append(
        f"Implementation match accuracy (kelas): {implementation_accuracy:.6f}"
    )
    report_lines.append(
        f"Mean absolute error (MAE) between prediction probabilities: {mae:.8f}\n"
    )

    report_lines.append("Test set metrics:")
    report_lines.append(
        f"  From scratch - Accuracy: {scratch_accuracy:.4f}, F1 Score (Macro): {scratch_f1:.4f}"
    )
    report_lines.append(
        f"  Keras        - Accuracy: {keras_accuracy:.4f}, F1 Score (Macro): {keras_f1:.4f}\n"
    )

    class_names = list(label_mapping.keys())

    if scratch_classes is not None:
        report_lines.append("From scratch - Classification Report:")
        report_lines.append(
            classification_report(
                test_labels,
                scratch_classes,
                target_names=class_names,
                zero_division=0,
            )
        )
        report_lines.append("\n")
    else:
        report_lines.append(
            "From scratch - Classification Report: Not available due to invalid predictions.\n"
        )

    report_lines.append("Keras - Classification Report:")
    report_lines.append(
        classification_report(
            test_labels, keras_classes, target_names=class_names, zero_division=0
        )
    )
    report_lines.append("\n")

    if scratch_classes is not None:
        cm_scratch_obj = confusion_matrix(test_labels, scratch_classes)
        report_lines.append("From scratch - Confusion Matrix:")
        report_lines.append(np.array2string(cm_scratch_obj, separator=", "))
        report_lines.append("\n")
    else:
        report_lines.append(
            "From scratch - Confusion Matrix: Not available due to invalid predictions.\n"
        )

    cm_keras_obj = confusion_matrix(test_labels, keras_classes)
    report_lines.append("Keras - Confusion Matrix:")
    report_lines.append(np.array2string(cm_keras_obj, separator=", "))
    report_lines.append("\n")

    # Cetak semua laporan detail ke konsol
    print("\n--- Laporan Detail Implementasi From Scratch ---")
    for line in report_lines:
        print(line)
    print("--- Akhir Laporan Detail ---")

    # Visualisasi hanya jika prediksi scratch valid
    if scratch_classes is not None and scratch_preds is not None:
        # Buat dan tampilkan confusion matrix
        plt.figure(figsize=(13, 5.5))

        plt.subplot(1, 2, 1)
        plt.imshow(cm_scratch_obj, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("From Scratch Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        for i in range(cm_scratch_obj.shape[0]):
            for j in range(cm_scratch_obj.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm_scratch_obj[i, j], "d"),
                    horizontalalignment="center",
                    color=(
                        "white"
                        if cm_scratch_obj[i, j] > cm_scratch_obj.max() / 2.0
                        else "black"
                    ),
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()  

        plt.subplot(1, 2, 2)
        plt.imshow(cm_keras_obj, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Keras Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        for i in range(cm_keras_obj.shape[0]):
            for j in range(cm_keras_obj.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm_keras_obj[i, j], "d"),
                    horizontalalignment="center",
                    color=(
                        "white"
                        if cm_keras_obj[i, j] > cm_keras_obj.max() / 2.0
                        else "black"
                    ),
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        plt.suptitle("Perbandingan Confusion Matrix", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.show()  
        plt.close()  

        # Bandingkan distribusi prediksi
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(np.max(scratch_preds, axis=1), bins=20, alpha=0.7, label="Scratch")
        plt.title("Scratch: Distribusi Probabilitas Maks")
        plt.xlabel("Probabilitas Maks")
        plt.ylabel("Jumlah")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(
            np.max(keras_preds, axis=1),
            bins=20,
            alpha=0.7,
            label="Keras",
            color="orange",
        )
        plt.title("Keras: Distribusi Probabilitas Maks")
        plt.xlabel("Probabilitas Maks")
        plt.ylabel("Jumlah")
        plt.legend()

        plt.suptitle("Perbandingan Distribusi Probabilitas Maksimum", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()  
        plt.close()  

        # Bandingkan prediksi individual
        plt.figure(figsize=(10, 6))
        sample_size = min(20, len(test_texts))
        if len(test_texts) > 0:
            sample_indices = np.random.choice(
                len(test_texts), sample_size, replace=False
            )
            if scratch_preds is not None and len(scratch_preds) == len(keras_preds):
                scratch_sample_preds = scratch_preds[sample_indices]
                keras_sample_preds = keras_preds[sample_indices]
                abs_diffs = np.abs(scratch_sample_preds - keras_sample_preds)
                mean_diffs_per_sample = np.mean(abs_diffs, axis=1)

                plt.bar(range(sample_size), mean_diffs_per_sample)
                plt.title("Perbedaan Absolut Rata-rata Probabilitas per Sampel")
                plt.xlabel("Indeks Sampel Acak")
                plt.ylabel("MAE Probabilitas")
                plt.xticks(
                    range(sample_size),
                    sample_indices.astype(str),
                    rotation=45,
                    ha="right",
                )
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show() 
            else:
                print(
                    "Tidak dapat memplot perbedaan prediksi karena scratch_preds tidak valid atau panjangnya tidak cocok."
                )
        else:
            print("Tidak ada data tes untuk memplot perbedaan prediksi individual.")
        plt.close() 
    else:
        print("Plotting dilewati karena prediksi scratch tidak valid.")

    return scratch_rnn
