import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import config


def create_rnn_model(
    vocab_size,
    embedding_dim=config.EMBEDDING_DIM,
    rnn_units=config.RNN_UNITS,
    num_rnn_layers=config.NUM_RNN_LAYERS,
    bidirectional=config.BIDIRECTIONAL,
    dropout_rate=config.DROPOUT_RATE,
    l2_reg=config.L2_REG,
    num_classes=3,
    learning_rate=config.LEARNING_RATE,
):
    model = models.Sequential(name="Simple_RNN_Classifier")

    # Layer Embedding - mengubah token jadi vektor, dengan regularisasi L2
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embedding_layer",
        )
    )

    # Dropout khusus untuk embedding - mencegah overfitting di representasi kata
    model.add(
        layers.SpatialDropout1D(config.EMBEDDING_DROPOUT, name="embedding_dropout")
    )

    # Tambahkan layer RNN sesuai jumlah yang diminta
    for i in range(num_rnn_layers):
        # Untuk layer terakhir, tidak perlu return sequences (kecuali layer terakhir)
        return_sequences = i < num_rnn_layers - 1

        # Buat layer SimpleRNN dengan regularisasi L2
        rnn_layer = layers.SimpleRNN(
            rnn_units,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            recurrent_dropout=config.RECURRENT_DROPOUT,  # Dropout internal pada koneksi recurrent
            name=f"simplernn_layer_{i+1}",
        )

        # Bungkus dengan Bidirectional jika diminta (baca dari dua arah)
        if bidirectional:
            model.add(layers.Bidirectional(rnn_layer, name=f"bidirectional_rnn_{i+1}"))
        else:
            model.add(rnn_layer)

        # Tambahkan dropout setelah tiap layer RNN
        model.add(layers.Dropout(dropout_rate, name=f"dropout_rnn_{i+1}"))

    # Layer output dengan softmax untuk klasifikasi multiclass
    model.add(
        layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            name="output_dense_layer",
        )
    )

    # Compile model dengan gradient clipping (mencegah gradien meledak)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",  # Untuk label kategorikal dalam bentuk integer
        metrics=["accuracy"],
    )

    return model
