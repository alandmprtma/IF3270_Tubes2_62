# Parameter pemrosesan data (umumnya sama dengan RNN)
MAX_TOKENS = 8000
OUTPUT_SEQ_LEN = 80

# Parameter training (umumnya sama dengan RNN)
BATCH_SIZE = 24
EPOCHS = 40 # Mungkin perlu disesuaikan untuk LSTM
RANDOM_SEED = 42

# Arsitektur model LSTM default
EMBEDDING_DIM = 96
LSTM_UNITS = 64  # Jumlah unit LSTM - nilai default (mungkin berbeda dari RNN_UNITS)
NUM_LSTM_LAYERS = 1  # Jumlah layer LSTM - nilai default
BIDIRECTIONAL_LSTM = True # Spesifik untuk LSTM jika ingin dibedakan

# Parameter regularisasi (umumnya sama dengan RNN)
DROPOUT_RATE = 0.3
EMBEDDING_DROPOUT = 0.2
RECURRENT_DROPOUT_LSTM = 0.1 # Dropout internal di LSTM
L2_REG = 0.01

# Parameter optimisasi (umumnya sama dengan RNN)
LEARNING_RATE = 0.001
LR_FACTOR = 0.7
LR_PATIENCE = 3
MIN_LR = 0.00001

# Parameter early stopping (umumnya sama dengan RNN)
ES_PATIENCE = 6