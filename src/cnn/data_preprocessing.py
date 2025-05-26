from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf

def compute_macro_f1_score(y_true, y_pred_probs):
    """Compute macro F1-score from predicted probabilities"""
    y_pred = np.argmax(y_pred_probs, axis=1)
    return f1_score(y_true, y_pred, average='macro')

# Load and preprocess CIFAR-10 data
def load_cifar10_data():
    """
    Load and preprocess CIFAR-10 dataset with train/validation/test split
    
    Returns:
    --------
    tuple: (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()
    
    # Split training data into train (40k) and validation (10k) sets
    split_idx = 40000
    x_train = x_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    x_val = x_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Test set: {x_test.shape}, {y_test.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names