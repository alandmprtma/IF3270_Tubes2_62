# Parameter pemrosesan data 
MAX_TOKENS = 8000
OUTPUT_SEQ_LEN = 80

# Parameter training 
BATCH_SIZE = 32
EPOCHS = 24
RANDOM_SEED = 42

# Arsitektur model LSTM default
EMBEDDING_DIM = 128
LSTM_UNITS = 64  
NUM_LSTM_LAYERS = 2  
BIDIRECTIONAL_LSTM = True

# Parameter regularisasi 
DROPOUT_RATE = 0.2
EMBEDDING_DROPOUT = 0.1
RECURRENT_DROPOUT_LSTM = 0.05 # Dropout internal di LSTM
L2_REG = 0.0001

# Parameter optimisasi 
LEARNING_RATE = 0.001
LR_FACTOR = 0.7
LR_PATIENCE = 3
MIN_LR = 0.00001

# Parameter early stopping
ES_PATIENCE = 6