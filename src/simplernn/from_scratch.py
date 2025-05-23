import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
import os

# Load local modules
from data_preprocessing import load_data, compute_f1_score

class SimpleRNNFromScratch:
    """
    Implementation of a SimpleRNN model from scratch using NumPy.
    This class loads weights from a trained Keras model and performs
    forward propagation to make predictions.
    """
    
    def __init__(self, keras_model_path, vectorizer_path):
        """
        Initialize the from-scratch SimpleRNN by loading weights from a pre-trained Keras model
        
        Parameters:
        -----------
        keras_model_path : str
            Path to the saved Keras model
        vectorizer_path : str
            Path to the saved text vectorizer
        """
        print(f"Loading Keras model from: {keras_model_path}")
        self.keras_model = tf.keras.models.load_model(keras_model_path)
        self.keras_model.summary()
        
        # Extract weights from each layer of the Keras model
        self.extract_weights_from_keras_model()
        
        # Load vectorizer
        print(f"Loading vectorizer from: {vectorizer_path}")
        self.vectorizer_model = tf.keras.models.load_model(vectorizer_path)
        self.vectorizer = self.vectorizer_model.layers[0]
        
        print("Model loaded and weights extracted successfully")
    
    def extract_weights_from_keras_model(self):
        """Extract weights from the loaded Keras model"""
        self.weights = {}
        
        # Get layer names from summary
        layer_names = [layer.name for layer in self.keras_model.layers]
        print(f"Layers in model: {layer_names}")
        
        # Find embedding layer
        embedding_layer = self.keras_model.get_layer("embedding_layer")
        self.weights["embedding"] = embedding_layer.get_weights()[0]  # Embedding matrix
        print(f"Embedding weights shape: {self.weights['embedding'].shape}")
        
        # Handle multiple bidirectional layers
        self.bidirectional_layers = []
        for i, layer_name in enumerate(layer_names):
            if "bidirectional" in layer_name:
                self.bidirectional_layers.append(layer_name)
        
        print(f"Found {len(self.bidirectional_layers)} bidirectional layers")
        
        # Extract weights for all bidirectional layers
        for i, layer_name in enumerate(self.bidirectional_layers):
            bidirectional_layer = self.keras_model.get_layer(layer_name)
            forward_layer = bidirectional_layer.forward_layer
            backward_layer = bidirectional_layer.backward_layer
            
            # Forward weights
            prefix = f"layer{i+1}_"  # Add layer number to weight names
            self.weights[prefix + "forward_rnn_kernel"] = forward_layer.get_weights()[0]
            self.weights[prefix + "forward_rnn_recurrent_kernel"] = forward_layer.get_weights()[1]
            self.weights[prefix + "forward_rnn_bias"] = forward_layer.get_weights()[2]
            
            # Backward weights
            self.weights[prefix + "backward_rnn_kernel"] = backward_layer.get_weights()[0]
            self.weights[prefix + "backward_rnn_recurrent_kernel"] = backward_layer.get_weights()[1]
            self.weights[prefix + "backward_rnn_bias"] = backward_layer.get_weights()[2]
            
            print(f"Layer {i+1} - Forward RNN kernel shape: {self.weights[prefix + 'forward_rnn_kernel'].shape}")
            print(f"Layer {i+1} - Forward RNN recurrent kernel shape: {self.weights[prefix + 'forward_rnn_recurrent_kernel'].shape}")
        
        # Extract output dense layer weights
        output_layer = self.keras_model.get_layer("output_dense_layer")
        self.weights["output_kernel"] = output_layer.get_weights()[0]
        self.weights["output_bias"] = output_layer.get_weights()[1]
        print(f"Output kernel shape: {self.weights['output_kernel'].shape}")
        print(f"Output bias shape: {self.weights['output_bias'].shape}")
    
    def embedding_forward(self, indices):
        """
        Embedding layer forward pass
        
        Parameters:
        -----------
        indices : numpy.ndarray
            Integer indices of shape (batch_size, sequence_length)
            
        Returns:
        --------
        numpy.ndarray
            Embedding vectors of shape (batch_size, sequence_length, embedding_dim)
        """
        # Lookup embedding vectors for each index
        # Each row i in indices gets mapped to the corresponding embedding vector
        return np.array([self.weights["embedding"][idx] for idx in indices])
    
    def simple_rnn_forward(self, inputs, kernel, recurrent_kernel, bias):
        """
        SimpleRNN cell forward pass (fixed)
        """
        batch_size, seq_length, input_dim = inputs.shape
        units = recurrent_kernel.shape[1]  # Note: Using shape[1] instead of shape[0]
        
        # Initialize hidden state with zeros
        h_t = np.zeros((batch_size, units))
        
        # Process each time step
        for t in range(seq_length):
            # Get input at current time step
            x_t = inputs[:, t, :]
            
            # Compute input projection
            input_projection = np.dot(x_t, kernel)
            
            # Compute recurrent projection
            recurrent_projection = np.dot(h_t, recurrent_kernel)
            
            # Debug: print shapes to verify alignment
            # print(f"Input proj shape: {input_projection.shape}, Recurrent proj shape: {recurrent_projection.shape}, Bias shape: {bias.shape}")
            
            # SimpleRNN step with explicit broadcasting of bias
            h_t = np.tanh(input_projection + recurrent_projection + np.reshape(bias, (1, -1)))
        
        return h_t
    
    def bidirectional_rnn_forward(self, inputs, layer_idx=1):
        """
        Bidirectional SimpleRNN forward pass for a specific layer
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input tensor
        layer_idx : int
            Layer number (1-based) to use
        
        Returns:
        --------
        numpy.ndarray
            Concatenated forward and backward hidden states
        """
        prefix = f"layer{layer_idx}_"
        
        # Forward pass
        forward_output = self.simple_rnn_forward(
            inputs, 
            self.weights[prefix + "forward_rnn_kernel"],
            self.weights[prefix + "forward_rnn_recurrent_kernel"],
            self.weights[prefix + "forward_rnn_bias"]
        )
        
        # Backward pass (reverse the input sequence)
        reversed_inputs = inputs.copy()
        reversed_inputs = reversed_inputs[:, ::-1, :]
        backward_output = self.simple_rnn_forward(
            reversed_inputs,
            self.weights[prefix + "backward_rnn_kernel"],
            self.weights[prefix + "backward_rnn_recurrent_kernel"],
            self.weights[prefix + "backward_rnn_bias"]
        )
        
        # Concatenate forward and backward outputs
        return np.concatenate([forward_output, backward_output], axis=1)
    
    def dense_forward(self, inputs, kernel, bias):
        """
        Dense layer forward pass with softmax activation
        
        Parameters:
        -----------
        inputs : numpy.ndarray
            Input tensor
        kernel : numpy.ndarray
            Weight matrix
        bias : numpy.ndarray
            Bias vector
            
        Returns:
        --------
        numpy.ndarray
            Output probabilities after softmax
        """
        # Linear transformation
        logits = np.dot(inputs, kernel) + bias
        
        # Softmax activation for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probabilities
    
    def forward(self, text_sequences):
        """
        Full model forward pass with multiple bidirectional layers
        """
        # Embedding layer
        embedded = self.embedding_forward(text_sequences)
        print(f"Embedded shape: {embedded.shape}, min: {embedded.min()}, max: {embedded.max()}, mean: {embedded.mean()}")
        
        # Process first bidirectional layer
        if len(self.bidirectional_layers) >= 1:
            # First layer expects sequences
            # For first layer, maintain the sequence dimension for next layer
            batch_size, seq_length, input_dim = embedded.shape
            units_per_direction = self.weights["layer1_forward_rnn_recurrent_kernel"].shape[0]
            
            # Process each time step for first bidirectional layer
            # For first layer we need to preserve sequence outputs
            forward_outputs = np.zeros((batch_size, seq_length, units_per_direction))
            backward_outputs = np.zeros((batch_size, seq_length, units_per_direction))
            
            # Forward direction
            h_t = np.zeros((batch_size, units_per_direction))
            for t in range(seq_length):
                x_t = embedded[:, t, :]
                h_t = np.tanh(
                    np.dot(x_t, self.weights["layer1_forward_rnn_kernel"]) +
                    np.dot(h_t, self.weights["layer1_forward_rnn_recurrent_kernel"]) + 
                    self.weights["layer1_forward_rnn_bias"]
                )
                forward_outputs[:, t, :] = h_t
            
            # Backward direction (process sequence in reverse)
            h_t = np.zeros((batch_size, units_per_direction))
            for t in range(seq_length-1, -1, -1):
                x_t = embedded[:, t, :]
                h_t = np.tanh(
                    np.dot(x_t, self.weights["layer1_backward_rnn_kernel"]) +
                    np.dot(h_t, self.weights["layer1_backward_rnn_recurrent_kernel"]) + 
                    self.weights["layer1_backward_rnn_bias"]
                )
                backward_outputs[:, t, :] = h_t
            
            # Concatenate forward and backward outputs along the last dimension
            first_layer_output = np.concatenate([forward_outputs, backward_outputs], axis=2)
            
            # Apply dropout (for completeness, though we can't simulate exact dropout)
            # In inference, dropout is not applied, so this is fine
            
            # Process second bidirectional layer (if present)
            if len(self.bidirectional_layers) >= 2:
                # Second layer consumes entire sequences and returns only final state
                second_layer_output = self.bidirectional_rnn_forward(first_layer_output, layer_idx=2)
                rnn_output = second_layer_output
            else:
                # If only one layer, use the final states
                rnn_output = np.concatenate([
                    forward_outputs[:, -1, :],  # Last time step of forward
                    backward_outputs[:, 0, :]   # First time step of backward (processed last)
                ], axis=1)
        else:
            # Fallback for non-bidirectional (shouldn't happen with your model)
            rnn_output = self.simple_rnn_forward(
                embedded,
                self.weights["rnn_kernel"],
                self.weights["rnn_recurrent_kernel"],
                self.weights["rnn_bias"]
            )
        
        print(f"RNN output shape: {rnn_output.shape}, min: {rnn_output.min()}, max: {rnn_output.max()}, mean: {rnn_output.mean()}")
        
        # Output layer with softmax
        logits = np.dot(rnn_output, self.weights["output_kernel"]) + self.weights["output_bias"]
        print(f"Logits: {logits[:2]}")  # Print first 2 examples
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Debug prediction diversity
        unique_predictions = np.unique(np.argmax(output, axis=1))
        print(f"Unique predicted classes: {unique_predictions}")
        
        return output
    
    def predict(self, texts):
        """
        Make predictions for text inputs
        
        Parameters:
        -----------
        texts : list or numpy.ndarray
            Text inputs
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities for each class
        """
        # Convert texts to sequences using the vectorizer
        sequences = self.vectorizer(texts).numpy()
        
        # Forward pass to get probabilities
        return self.forward(sequences)
    
    def compare_with_keras(self, texts):
        """
        Compare predictions from scratch implementation with Keras model
        
        Parameters:
        -----------
        texts : list or numpy.ndarray
            Text inputs
            
        Returns:
        --------
        tuple
            (from_scratch_predictions, keras_predictions, accuracy)
        """
        # Get predictions from our scratch implementation
        start_time_scratch = time.time()
        scratch_preds = self.predict(texts)
        scratch_time = time.time() - start_time_scratch
        
        # Get predictions from Keras
        start_time_keras = time.time()
        sequences = self.vectorizer(texts)
        keras_preds = self.keras_model.predict(sequences)
        keras_time = time.time() - start_time_keras
        
        # Calculate accuracy between the two implementations
        scratch_classes = np.argmax(scratch_preds, axis=1)
        keras_classes = np.argmax(keras_preds, axis=1)
        accuracy = np.mean(scratch_classes == keras_classes)
        
        print(f"\nPrediction time comparison:")
        print(f"From scratch: {scratch_time:.4f} seconds")
        print(f"Keras: {keras_time:.4f} seconds")
        print(f"Time ratio (scratch/keras): {scratch_time/keras_time:.2f}x")
        
        print(f"\nImplementation match accuracy: {accuracy:.4f}")
        
        # Calculate mean absolute error between predictions
        mae = np.mean(np.abs(scratch_preds - keras_preds))
        print(f"Mean absolute error between predictions: {mae:.6f}")
        
        return scratch_preds, keras_preds, accuracy

def run_from_scratch_comparison(model_path, vectorizer_path):
    """
    Run a complete evaluation of the from-scratch implementation
    
    Parameters:
    -----------
    model_path : str
        Path to the saved Keras model
    vectorizer_path : str
        Path to the saved text vectorizer
        
    Returns:
    --------
    SimpleRNNFromScratch
        The from-scratch implementation object
    """
    print("\n" + "="*50)
    print("SIMPLE RNN FROM SCRATCH IMPLEMENTATION")
    print("="*50)
    
    # Create output directory
    output_dir = "from_scratch_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    (train_texts, train_labels), (valid_texts, valid_labels), (test_texts, test_labels), label_mapping, num_classes = load_data()
    print(f"Loaded test data: {len(test_texts)} samples")
    
    # Initialize from-scratch implementation
    scratch_rnn = SimpleRNNFromScratch(model_path, vectorizer_path)
    
    # Compare implementations on test data
    print("\nComparing implementations on test data...")
    scratch_preds, keras_preds, implementation_accuracy = scratch_rnn.compare_with_keras(test_texts)
    
    # Get class predictions
    scratch_classes = np.argmax(scratch_preds, axis=1)
    keras_classes = np.argmax(keras_preds, axis=1)
    
    # Calculate metrics for both implementations
    scratch_accuracy = np.mean(scratch_classes == test_labels)
    keras_accuracy = np.mean(keras_classes == test_labels)
    
    scratch_f1 = compute_f1_score(test_labels, scratch_preds)
    keras_f1 = compute_f1_score(test_labels, keras_preds)
    
    print("\nTest set metrics:")
    print(f"From scratch - Accuracy: {scratch_accuracy:.4f}, F1 Score: {scratch_f1:.4f}")
    print(f"Keras - Accuracy: {keras_accuracy:.4f}, F1 Score: {keras_f1:.4f}")
    
    # Display classification reports
    class_names = list(label_mapping.keys())
    
    print("\nFrom scratch - Classification Report:")
    print(classification_report(test_labels, scratch_classes, target_names=class_names))
    
    print("\nKeras - Classification Report:")
    print(classification_report(test_labels, keras_classes, target_names=class_names))
    
    # Create and save confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm_scratch = confusion_matrix(test_labels, scratch_classes)
    plt.imshow(cm_scratch, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('From Scratch Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.subplot(1, 2, 2)
    cm_keras = confusion_matrix(test_labels, keras_classes)
    plt.imshow(cm_keras, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Keras Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices_comparison.png")
    
    # Compare prediction distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(np.max(scratch_preds, axis=1), bins=20, alpha=0.7)
    plt.title('From Scratch: Max Probability Distribution')
    plt.xlabel('Max Probability')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.max(keras_preds, axis=1), bins=20, alpha=0.7)
    plt.title('Keras: Max Probability Distribution')
    plt.xlabel('Max Probability')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_distributions.png")
    
    # Compare individual predictions
    plt.figure(figsize=(10, 6))
    
    # Get some sample indices
    sample_size = min(20, len(test_texts))
    sample_indices = np.random.choice(len(test_texts), sample_size, replace=False)
    
    # Get probabilities for each class for scratch and keras
    scratch_sample_preds = scratch_preds[sample_indices]
    keras_sample_preds = keras_preds[sample_indices]
    
    # Calculate absolute differences between implementations
    abs_diffs = np.abs(scratch_sample_preds - keras_sample_preds)
    mean_diffs = np.mean(abs_diffs, axis=1)
    
    plt.bar(range(sample_size), mean_diffs)
    plt.title('Mean Absolute Difference in Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Absolute Difference')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_differences.png")
    
    # Save detailed results to text file
    with open(f"{output_dir}/detailed_results.txt", "w") as f:
        f.write("SIMPLE RNN FROM SCRATCH EVALUATION\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Implementation match accuracy: {implementation_accuracy:.4f}\n")
        f.write(f"Mean absolute error between predictions: {np.mean(np.abs(scratch_preds - keras_preds)):.6f}\n\n")
        
        f.write("Test set metrics:\n")
        f.write(f"From scratch - Accuracy: {scratch_accuracy:.4f}, F1 Score: {scratch_f1:.4f}\n")
        f.write(f"Keras - Accuracy: {keras_accuracy:.4f}, F1 Score: {keras_f1:.4f}\n\n")
        
        f.write("From scratch - Classification Report:\n")
        f.write(classification_report(test_labels, scratch_classes, target_names=class_names))
        f.write("\n")
        
        f.write("Keras - Classification Report:\n")
        f.write(classification_report(test_labels, keras_classes, target_names=class_names))
        f.write("\n")
        
        f.write("From scratch - Confusion Matrix:\n")
        f.write(str(cm_scratch))
        f.write("\n\n")
        
        f.write("Keras - Confusion Matrix:\n")
        f.write(str(cm_keras))
    
    return scratch_rnn

if __name__ == "__main__":
    # Specify which model to load
    model_path = "models/optimal_rnn_l2_u96_bi_full_model.keras"
    vectorizer_path = "models/optimal_rnn_l2_u96_bi_vectorizer.keras"
    
    # Run the from-scratch comparison
    scratch_rnn = run_from_scratch_comparison(model_path, vectorizer_path)
    
    # Example of using the from-scratch implementation for a single prediction
    sample_texts = [
        "Film ini sangat bagus, saya sangat menikmatinya!",
        "Pelayanan di hotel ini mengecewakan dan kamarnya kotor.",
        "Harga produk ini cukup standar, tidak mahal tidak murah."
    ]
    
    print("\nExample predictions from scratch implementation:")
    predictions = scratch_rnn.predict(sample_texts)
    
    sentiment_classes = ["negative", "neutral", "positive"]
    for i, text in enumerate(sample_texts):
        pred_class = np.argmax(predictions[i])
        pred_prob = predictions[i][pred_class]
        print(f"\nText: {text}")
        print(f"Predicted sentiment: {sentiment_classes[pred_class]} with {pred_prob:.4f} probability")
        print(f"Full probabilities: negative={predictions[i][0]:.4f}, neutral={predictions[i][1]:.4f}, positive={predictions[i][2]:.4f}")