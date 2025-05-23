import os
import numpy as np
import tensorflow as tf

# Create directories for outputs
os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Step 1: Training base model")
from train import train_and_evaluate_model
model, history, _, _, _ = train_and_evaluate_model()

print("\nStep 2: Running hyperparameter experiments")
from experiment import experiment_rnn_layers, experiment_rnn_units, experiment_bidirectional
experiment_rnn_layers()
experiment_rnn_units()
experiment_bidirectional()

print("\nStep 3: Implementing forward propagation from scratch")
from from_scratch import compare_with_keras
compare_with_keras()

print("\nAll tasks completed successfully!")