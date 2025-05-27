import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Fungsi untuk melatih dan mengevaluasi model
def train_and_evaluate_cnn_model(model, train_dataset, valid_dataset, test_dataset, class_names, epochs=10, patience=5):
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience // 2, min_lr=1e-6)
    ]
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Print model summary
    model.summary()

    # Train model
    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks, verbose=1)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions and true labels
    y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)
    y_pred_probs = np.concatenate([model.predict(x) for x, _ in test_dataset], axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average="macro")
    
    return model, history, y_pred_probs, y_true, {"test_loss": test_loss, "test_accuracy": test_accuracy, "test_f1": f1}

# Fungsi untuk plot confusion matrix
def plot_confusion_matrix(preds, labels, class_names):
    cm = confusion_matrix(labels, np.argmax(preds, axis=1))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Model CNN', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=10)
    
    plt.tight_layout()
    plt.show()