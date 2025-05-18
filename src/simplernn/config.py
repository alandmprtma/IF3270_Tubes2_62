# Data processing parameters
MAX_TOKENS = 8000
OUTPUT_SEQ_LEN = 80

# Training parameters
BATCH_SIZE = 24
EPOCHS = 40
RANDOM_SEED = 42

# Default model architecture
EMBEDDING_DIM = 96
RNN_UNITS = 48
NUM_RNN_LAYERS = 1
BIDIRECTIONAL = True

# Regularization parameters
DROPOUT_RATE = 0.3  # Main dropout after RNN
EMBEDDING_DROPOUT = 0.2  # Dropout after embedding
RECURRENT_DROPOUT = 0.1  # Internal RNN state dropout
L2_REG = 0.001  # L2 regularization factor

# Optimization parameters
LEARNING_RATE = 0.001
LR_FACTOR = 0.7  # Factor for ReduceLROnPlateau
LR_PATIENCE = 3  # Patience for ReduceLROnPlateau
MIN_LR = 0.00001

# Early stopping parameters
ES_PATIENCE = 6

# Experiment variations - for testing different configurations
RNN_LAYERS_VARIATIONS = [1, 2, 3]  # For layer count experiments
RNN_UNITS_VARIATIONS = [24, 48, 96]  # For unit count experiments
BIDIRECTIONAL_VARIATIONS = [False, True]  # For directionality experiments