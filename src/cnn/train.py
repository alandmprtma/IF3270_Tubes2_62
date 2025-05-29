import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from data_preprocessing import compute_macro_f1_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import config
import pandas as pd
import seaborn as sns

# Fungsi untuk melatih dan mengevaluasi model
def train_and_evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, 
                           model_name, epochs=10):
    """
    Train and evaluate a CNN model
    
    Returns:
    --------
    tuple: (model, history, test_f1_score)
    """
    print(f"\nTraining {model_name}...")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
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
    
    # Train model with callbacks
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading best weights from {checkpoint_path}")
        model.load_weights(checkpoint_path)
    
    # Evaluate on test set
    test_predictions = model.predict(x_test)
    test_f1 = compute_macro_f1_score(y_test, test_predictions)
    
    print(f"{model_name} - Test Macro F1-Score: {test_f1:.4f}")
    
    return model, history, test_f1

def plot_individual_training_history(history, model_name):
    """Plot training and validation curves for individual model"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linestyle='--')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_combined_comparison(histories, model_names, f1_scores, train_times):
    """Plot combined comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Loss comparison
    axes[0, 0].set_title('Training Loss Comparison')
    for history, name in zip(histories, model_names):
        axes[0, 0].plot(history.history['loss'], label=f'{name}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Validation loss comparison
    axes[0, 1].set_title('Validation Loss Comparison')
    for history, name in zip(histories, model_names):
        axes[0, 1].plot(history.history['val_loss'], label=f'{name}', linestyle='--')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1-Score comparison
    axes[1, 0].bar(model_names, f1_scores, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim(0, 1)
    for i, score in enumerate(f1_scores):
        axes[1, 0].text(i, score + 0.01, f'{score:.4f}', ha='center')
    
    # 4. Training time comparison
    axes[1, 1].bar(model_names, train_times, color=['orange', 'purple', 'brown'])
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    for i, time_val in enumerate(train_times):
        axes[1, 1].text(i, time_val + max(train_times) * 0.01, f'{time_val:.1f}s', ha='center')
    
    plt.tight_layout()
    plt.show()

def save_results_to_csv_conv(model_names, accuracies, f1_scores, losses, train_times, save_path):
    """Save experiment results to CSV"""
    num_layers = [int(name.split('_')[0]) for name in model_names]
    
    results_df = pd.DataFrame({
        'num_layers': num_layers,
        'accuracy': accuracies,
        'f1_score': f1_scores,
        'loss': losses,
        'train_time': train_times
    })
    
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    return results_df

def save_results_to_csv_filters(model_names, accuracies, f1_scores, losses, train_times, save_path):
    """Save filter experiment results to CSV"""
    filter_configs = []
    for name in model_names:
        parts = name.split('_')
        config = f"{parts[1]}-{parts[2]}-{parts[3]}"
        filter_configs.append(config)
    
    results_df = pd.DataFrame({
        'filter_config': filter_configs,
        'accuracy': accuracies,
        'f1_score': f1_scores,
        'loss': losses,
        'train_time': train_times
    })
    
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    return results_df

def save_results_to_csv_kernels(model_names, accuracies, f1_scores, losses, train_times, filename):
    """Save kernel sizes experiment results to CSV"""
    import pandas as pd
    
    kernel_configs = []
    for name in model_names:
        kernels_part = name.split('kernels_')[1]
        kernel_configs.append(kernels_part)
    
    results_dict = {
        'Model_Name': model_names,
        'Kernel_Configuration': kernel_configs,
        'Test_Accuracy': [f"{acc:.4f}" for acc in accuracies],
        'F1_Score': [f"{f1:.4f}" for f1 in f1_scores],
        'Test_Loss': [f"{loss:.4f}" for loss in losses],
        'Training_Time_seconds': [f"{time:.2f}" for time in train_times]
    }
    
    df = pd.DataFrame(results_dict)
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    
    return df

def save_results_to_csv_pooling(model_names, accuracies, f1_scores, losses, train_times, filename):
    """Save pooling types experiment results to CSV"""
    import pandas as pd
    
    pooling_types = []
    for name in model_names:
        # "max_pooling" atau "average_pooling"
        pooling_type = name.split('_pooling')[0]
        pooling_types.append(pooling_type.capitalize())
    
    results_dict = {
        'Model_Name': model_names,
        'Pooling_Type': pooling_types,
        'Test_Accuracy': [f"{acc:.4f}" for acc in accuracies],
        'F1_Score': [f"{f1:.4f}" for f1 in f1_scores],
        'Test_Loss': [f"{loss:.4f}" for loss in losses],
        'Training_Time_seconds': [f"{time:.2f}" for time in train_times]
    }
    
    df = pd.DataFrame(results_dict)
    df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    
    return df