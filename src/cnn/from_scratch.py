import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score as sklearn_f1_score,
)
from data_preprocessing import (
    load_cifar10_data,
    compute_macro_f1_score
)

class CNNFromScratch:
    """
    Implementation of a CNN model from scratch using NumPy.
    This class loads weights from a trained Keras model and performs
    forward propagation to make predictions.
    """
    
    def __init__(self, keras_model_path):
        """
        Initialize the from-scratch CNN by loading weights from a pre-trained Keras model
        
        Parameters:
        -----------
        keras_model_path : str
            Path to the saved Keras model
        """
        print(f"Loading Keras model from: {keras_model_path}")
        self.keras_model = tf.keras.models.load_model(keras_model_path)
        self.keras_model.summary()
        
        # Extract weights from each layer of the Keras model
        self.extract_weights_from_keras_model()
        print("Model loaded and weights extracted successfully")
    
    def extract_weights_from_keras_model(self):
        """Extract weights from the loaded Keras model"""
        self.weights = {}
        self.layer_info = {}
        
        for layer in self.keras_model.layers:
            layer_name = layer.name
            layer_type = type(layer).__name__
            
            print(f"Processing layer: {layer_name} ({layer_type})")
            
            if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                layer_weights = layer.get_weights()
                
                if layer_type == 'Conv2D':
                    # Conv2D layer: [kernel, bias]
                    self.weights[f"{layer_name}_kernel"] = layer_weights[0]
                    self.weights[f"{layer_name}_bias"] = layer_weights[1]
                    
                    # Store layer configuration
                    self.layer_info[layer_name] = {
                        'type': 'Conv2D',
                        'filters': layer.filters,
                        'kernel_size': layer.kernel_size,
                        'strides': layer.strides,
                        'padding': layer.padding,
                        'activation': layer.activation.__name__
                    }
                    
                    print(f"  Kernel shape: {layer_weights[0].shape}")
                    print(f"  Bias shape: {layer_weights[1].shape}")
                
                elif layer_type == 'Dense':
                    # Dense layer: [kernel, bias]
                    self.weights[f"{layer_name}_kernel"] = layer_weights[0]
                    self.weights[f"{layer_name}_bias"] = layer_weights[1]
                    
                    self.layer_info[layer_name] = {
                        'type': 'Dense',
                        'units': layer.units,
                        'activation': layer.activation.__name__
                    }
                    
                    print(f"  Kernel shape: {layer_weights[0].shape}")
                    print(f"  Bias shape: {layer_weights[1].shape}")
            
            elif layer_type in ['MaxPooling2D', 'AveragePooling2D']:
                # Pooling layers don't have weights but we need their config
                self.layer_info[layer_name] = {
                    'type': layer_type,
                    'pool_size': layer.pool_size,
                    'strides': layer.strides,
                    'padding': layer.padding
                }
            
            elif layer_type == 'Flatten':
                self.layer_info[layer_name] = {'type': 'Flatten'}
    
    def conv2d_forward(self, inputs, kernel, bias, strides=(1, 1), padding='valid', activation='relu'):
        """
        2D Convolution forward pass
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input tensor of shape (batch_size, height, width, channels)
        kernel : numpy.ndarray
            Convolution kernel of shape (kernel_height, kernel_width, input_channels, output_channels)
        bias : numpy.ndarray
            Bias vector of shape (output_channels,)
        strides : tuple
            Stride for convolution
        padding : str
            Padding type ('valid' or 'same')
        activation : str
            Activation function name
        
        Returns:
        --------
        numpy.ndarray
            Output tensor after convolution
        """
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width, _, output_channels = kernel.shape
        stride_h, stride_w = strides
        
        # Calculate output dimensions
        if padding == 'valid':
            output_height = (input_height - kernel_height) // stride_h + 1
            output_width = (input_width - kernel_width) // stride_w + 1
            pad_h = pad_w = 0
        else:  # padding == 'same'
            output_height = input_height // stride_h
            output_width = input_width // stride_w
            pad_h = max(0, (output_height - 1) * stride_h + kernel_height - input_height)
            pad_w = max(0, (output_width - 1) * stride_w + kernel_width - input_width)
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            inputs = np.pad(inputs, 
                          ((0, 0), (pad_h//2, pad_h - pad_h//2), 
                           (pad_w//2, pad_w - pad_w//2), (0, 0)), 
                          mode='constant')
        
        # Initialize output
        output = np.zeros((batch_size, output_height, output_width, output_channels))
        
        # Perform convolution
        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    h_start = h * stride_h
                    h_end = h_start + kernel_height
                    w_start = w * stride_w
                    w_end = w_start + kernel_width
                    
                    # Extract input patch
                    input_patch = inputs[b, h_start:h_end, w_start:w_end, :]
                    
                    # Convolution operation
                    for f in range(output_channels):
                        output[b, h, w, f] = np.sum(input_patch * kernel[:, :, :, f]) + bias[f]
        
        # Apply activation
        if activation == 'relu':
            output = np.maximum(0, output)
        elif activation == 'softmax':
            # Apply softmax along the last dimension
            exp_output = np.exp(output - np.max(output, axis=-1, keepdims=True))
            output = exp_output / np.sum(exp_output, axis=-1, keepdims=True)
        
        return output
    
    def pooling_forward(self, inputs, pool_size=(2, 2), strides=(2, 2), pooling_type='max'):
        """
        Pooling layer forward pass
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input tensor of shape (batch_size, height, width, channels)
        pool_size : tuple
            Size of pooling window
        strides : tuple
            Stride for pooling
        pooling_type : str
            Type of pooling ('max' or 'average')
        
        Returns:
        --------
        numpy.ndarray
            Output tensor after pooling
        """
        batch_size, input_height, input_width, channels = inputs.shape
        pool_h, pool_w = pool_size
        stride_h, stride_w = strides
        
        # Calculate output dimensions
        output_height = (input_height - pool_h) // stride_h + 1
        output_width = (input_width - pool_w) // stride_w + 1
        
        # Initialize output
        output = np.zeros((batch_size, output_height, output_width, channels))
        
        # Perform pooling
        for b in range(batch_size):
            for h in range(output_height):
                for w in range(output_width):
                    h_start = h * stride_h
                    h_end = h_start + pool_h
                    w_start = w * stride_w
                    w_end = w_start + pool_w
                    
                    # Extract input patch
                    input_patch = inputs[b, h_start:h_end, w_start:w_end, :]
                    
                    # Apply pooling operation
                    if pooling_type == 'max':
                        output[b, h, w, :] = np.max(input_patch, axis=(0, 1))
                    else:  # average pooling
                        output[b, h, w, :] = np.mean(input_patch, axis=(0, 1))
        
        return output
    
    def flatten_forward(self, inputs):
        """
        Flatten layer forward pass
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input tensor
        
        Returns:
        --------
        numpy.ndarray
            Flattened tensor of shape (batch_size, flattened_size)
        """
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)
    
    def dense_forward(self, inputs, kernel, bias, activation='relu'):
        """
        Dense layer forward pass
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input tensor
        kernel : numpy.ndarray
            Weight matrix
        bias : numpy.ndarray
            Bias vector
        activation : str
            Activation function name
        
        Returns:
        --------
        numpy.ndarray
            Output tensor after dense layer
        """
        # Linear transformation
        output = np.dot(inputs, kernel) + bias
        
        # Apply activation
        if activation == 'relu':
            output = np.maximum(0, output)
        elif activation == 'softmax':
            # Softmax activation for numerical stability
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        
        return output
    
    def forward(self, inputs):
        """
        Full model forward pass
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input tensor of shape (batch_size, height, width, channels)
        
        Returns:
        --------
        numpy.ndarray
            Output probabilities
        """
        x = inputs.copy()
        
        # Process each layer in order
        for layer in self.keras_model.layers:
            layer_name = layer.name
            layer_info = self.layer_info.get(layer_name, {})
            layer_type = layer_info.get('type', '')
            
            if layer_type == 'Conv2D':
                kernel = self.weights[f"{layer_name}_kernel"]
                bias = self.weights[f"{layer_name}_bias"]
                activation = layer_info['activation']
                strides = layer_info['strides']
                padding = layer_info['padding']
                
                x = self.conv2d_forward(x, kernel, bias, strides, padding, activation)
                print(f"After {layer_name}: {x.shape}")
            
            elif layer_type in ['MaxPooling2D', 'AveragePooling2D']:
                pool_size = layer_info['pool_size']
                strides = layer_info['strides']
                pooling_type = 'max' if layer_type == 'MaxPooling2D' else 'average'
                
                x = self.pooling_forward(x, pool_size, strides, pooling_type)
                print(f"After {layer_name}: {x.shape}")
            
            elif layer_type == 'Flatten':
                x = self.flatten_forward(x)
                print(f"After {layer_name}: {x.shape}")
            
            elif layer_type == 'Dense':
                kernel = self.weights[f"{layer_name}_kernel"]
                bias = self.weights[f"{layer_name}_bias"]
                activation = layer_info['activation']
                
                x = self.dense_forward(x, kernel, bias, activation)
                print(f"After {layer_name}: {x.shape}")
        
        return x
    
    def predict(self, inputs):
        """
        Make predictions for input data
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input images
        
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities for each class
        """
        return self.forward(inputs)
    
    def compare_with_keras(self, test_data, test_labels):
        """
        Compare predictions from scratch implementation with Keras model
        
        Parameters:
        -----------
        test_data : numpy.ndarray
            Test images
        test_labels : numpy.ndarray
            Test labels
        
        Returns:
        --------
        tuple
            (from_scratch_predictions, keras_predictions, accuracy)
        """
        # Get predictions from our scratch implementation
        print("Getting predictions from scratch implementation...")
        start_time_scratch = time.time()
        scratch_preds = self.predict(test_data)
        scratch_time = time.time() - start_time_scratch
        
        # Get predictions from Keras
        print("Getting predictions from Keras model...")
        start_time_keras = time.time()
        keras_preds = self.keras_model.predict(test_data)
        keras_time = time.time() - start_time_keras
        
        # Calculate accuracy between the two implementations
        scratch_classes = np.argmax(scratch_preds, axis=1)
        keras_classes = np.argmax(keras_preds, axis=1)
        implementation_accuracy = np.mean(scratch_classes == keras_classes)
        
        # Calculate F1 scores
        scratch_f1 = compute_macro_f1_score(test_labels, scratch_preds)
        keras_f1 = compute_macro_f1_score(test_labels, keras_preds)
        
        print(f"\nPrediction time comparison:")
        print(f"From scratch: {scratch_time:.4f} seconds")
        print(f"Keras: {keras_time:.4f} seconds")
        print(f"Time ratio (scratch/keras): {scratch_time/keras_time:.2f}x")
        
        print(f"\nImplementation match accuracy: {implementation_accuracy:.4f}")
        print(f"Mean absolute error between predictions: {np.mean(np.abs(scratch_preds - keras_preds)):.6f}")
        
        print(f"\nTest set metrics:")
        print(f"From scratch - Macro F1-Score: {scratch_f1:.4f}")
        print(f"Keras - Macro F1-Score: {keras_f1:.4f}")
        
        return scratch_preds, keras_preds, implementation_accuracy
    

def run_from_scratch_comparison(model_path):
    """
    Run comparison between CNN from scratch implementation and Keras model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved Keras CNN model
    """
    print("\n" + "=" * 50)
    print("CNN FROM SCRATCH IMPLEMENTATION")
    print("=" * 50)

    # Load data - sesuaikan dengan fungsi load data untuk CNN/image
    # Asumsi: fungsi load_cifar10_data() mengembalikan data dalam format yang sesuai untuk CNN
    try:
        (
            (train_images, train_labels),
            (valid_images, valid_labels),
            (test_images, test_labels),
            label_mapping,
            num_classes,
        ) = load_cifar10_data()  # Sesuaikan nama fungsi jika berbeda
        print(f"Loaded test data: {len(test_images)} samples")
        print(f"Image shape: {test_images.shape[1:]}")
        print(f"Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Initialize from-scratch implementation
    try:
        scratch_cnn = CNNFromScratch(model_path)
    except Exception as e:
        print(f"Error initializing CNN from scratch: {e}")
        return None

    # Compare implementations on test data
    print("\nComparing implementations on test data...")

    try:
        scratch_preds, keras_preds, implementation_accuracy = (
            scratch_cnn.compare_with_keras(test_images, test_labels)
        )
    except Exception as e:
        print(f"Error during comparison: {e}")
        return None

    # Get predicted classes
    scratch_classes = None
    mae = 0.0
    
    if scratch_preds is not None and not (
        np.any(np.isnan(scratch_preds)) or np.any(np.isinf(scratch_preds))
    ):
        scratch_classes = np.argmax(scratch_preds, axis=1)
        mae = np.mean(np.abs(scratch_preds - keras_preds))
    else:
        print("WARNING: NaN or Inf found in scratch predictions!")
        mae = float("inf")

    keras_classes = np.argmax(keras_preds, axis=1)

    # Calculate metrics for both implementations
    scratch_accuracy = 0.0
    scratch_f1 = 0.0
    if scratch_classes is not None:
        scratch_accuracy = np.mean(scratch_classes == test_labels)
        scratch_f1 = compute_macro_f1_score(test_labels, scratch_preds)
    else:
        print("Warning: Scratch metrics cannot be calculated due to invalid predictions (NaN/Inf).")

    keras_accuracy = np.mean(keras_classes == test_labels)
    keras_f1 = compute_macro_f1_score(test_labels, keras_preds)

    # --- Start Detailed Report ---
    report_lines = []
    report_lines.append("CNN FROM SCRATCH EVALUATION")
    report_lines.append("=" * 50 + "\n")
    report_lines.append(f"Keras Model Path: {model_path}\n")

    report_lines.append(
        f"Implementation match accuracy (classes): {implementation_accuracy:.6f}"
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

    class_names = list(label_mapping.keys()) if isinstance(label_mapping, dict) else [f"Class_{i}" for i in range(num_classes)]

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

    # Print all detailed reports to console
    print("\n--- Detailed From Scratch Implementation Report ---")
    for line in report_lines:
        print(line)
    print("--- End of Detailed Report ---")

    # Visualization only if scratch predictions are valid
    if scratch_classes is not None and scratch_preds is not None:
        # Create and display confusion matrices
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

        plt.suptitle("Confusion Matrix Comparison", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close()

        # Compare prediction distributions
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(np.max(scratch_preds, axis=1), bins=20, alpha=0.7, label="Scratch")
        plt.title("Scratch: Max Probability Distribution")
        plt.xlabel("Max Probability")
        plt.ylabel("Count")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(
            np.max(keras_preds, axis=1),
            bins=20,
            alpha=0.7,
            label="Keras",
            color="orange",
        )
        plt.title("Keras: Max Probability Distribution")
        plt.xlabel("Max Probability")
        plt.ylabel("Count")
        plt.legend()

        plt.suptitle("Maximum Probability Distribution Comparison", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close()

        # Compare individual predictions
        plt.figure(figsize=(10, 6))
        sample_size = min(20, len(test_images))
        if len(test_images) > 0:
            sample_indices = np.random.choice(
                len(test_images), sample_size, replace=False
            )
            if scratch_preds is not None and len(scratch_preds) == len(keras_preds):
                scratch_sample_preds = scratch_preds[sample_indices]
                keras_sample_preds = keras_preds[sample_indices]
                abs_diffs = np.abs(scratch_sample_preds - keras_sample_preds)
                mean_diffs_per_sample = np.mean(abs_diffs, axis=1)

                plt.bar(range(sample_size), mean_diffs_per_sample)
                plt.title("Average Absolute Probability Difference per Sample")
                plt.xlabel("Random Sample Index")
                plt.ylabel("MAE Probability")
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
                    "Cannot plot prediction differences because scratch_preds is invalid or length mismatch."
                )
        else:
            print("No test data available to plot individual prediction differences.")
        plt.close()

        # Additional visualization: Show some sample images with predictions
        plt.figure(figsize=(15, 8))
        sample_size = min(8, len(test_images))
        sample_indices = np.random.choice(len(test_images), sample_size, replace=False)
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 4, i + 1)
            
            # Display image (assuming it's in a format suitable for display)
            if len(test_images.shape) == 4:  # (batch, height, width, channels)
                img = test_images[idx]
                if img.shape[-1] == 1:  # Grayscale
                    plt.imshow(img.squeeze(), cmap='gray')
                else:  # RGB
                    plt.imshow(img)
            else:
                plt.imshow(test_images[idx], cmap='gray')
            
            true_label = class_names[test_labels[idx]]
            scratch_pred = class_names[scratch_classes[idx]]
            keras_pred = class_names[keras_classes[idx]]
            
            title = f"True: {true_label}\nScratch: {scratch_pred}\nKeras: {keras_pred}"
            plt.title(title, fontsize=8)
            plt.axis('off')
        
        plt.suptitle("Sample Predictions Comparison", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close()
        
    else:
        print("Plotting skipped due to invalid scratch predictions.")

    return scratch_cnn