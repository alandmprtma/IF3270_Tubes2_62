# Parameter pemrosesan data
MAX_TOKENS = 8000  # Jumlah maksimum token yang dikenali model
OUTPUT_SEQ_LEN = (
    80  # Panjang maksimum sequence input (kalimat dipotong/dipadding ke ukuran ini)
)

# Parameter training
BATCH_SIZE = 24
EPOCHS = 40
RANDOM_SEED = 42

# Arsitektur model default
EMBEDDING_DIM = 96  # Dimensi vektor embedding
RNN_UNITS = 48  # Jumlah unit RNN - nilai default
NUM_RNN_LAYERS = 1  # Jumlah layer RNN - nilai default
BIDIRECTIONAL = True

# Parameter regularisasi - untuk mencegah overfitting
DROPOUT_RATE = 0.3  # Dropout setelah layer RNN
EMBEDDING_DROPOUT = 0.2  # Dropout setelah embedding, mencegah overfitting di awal
RECURRENT_DROPOUT = 0.1  # Dropout internal di RNN
L2_REG = 0.01  # Regularisasi L2, membatasi bobot agar tidak terlalu besar

# Parameter optimisasi
LEARNING_RATE = 0.001  # Learning rate awal Adam
LR_FACTOR = 0.7  # Faktor untuk mengurangi learning rate jika stuck
LR_PATIENCE = 3  # Tunggu 3 epoch dulu sebelum kurangi learning rate
MIN_LR = 0.00001  # Learning rate minimum

# Parameter early stopping
ES_PATIENCE = 6  # Berhenti training kalau 6 epoch tidak ada improvement

# Variasi untuk eksperimen - buat menguji konfigurasi berbeda
RNN_LAYERS_VARIATIONS = [1, 2, 3]  # Variasi jumlah layer
RNN_UNITS_VARIATIONS = [24, 48, 96]  # Variasi jumlah unit (neuron) per layer
BIDIRECTIONAL_VARIATIONS = [False, True]  # Variasi arah RNN (satu arah atau dua arah)
