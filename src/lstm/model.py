import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import config as config


def create_lstm_model(
    vocab_size,
    embedding_dim=config.EMBEDDING_DIM,
    lstm_units=config.LSTM_UNITS,
    num_lstm_layers=config.NUM_LSTM_LAYERS,
    bidirectional=config.BIDIRECTIONAL_LSTM,  
    dropout_rate=config.DROPOUT_RATE,
    embedding_dropout_rate=config.EMBEDDING_DROPOUT,
    recurrent_dropout_rate=config.RECURRENT_DROPOUT_LSTM,  
    l2_reg_strength=config.L2_REG,
    num_classes=3,  
    learning_rate=config.LEARNING_RATE,
):
    model = models.Sequential(name="LSTM_Classifier")

    # 1. Embedding Layer
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_regularizer=l2(l2_reg_strength),
            name="embedding_layer",
        )
    )
    model.add(layers.SpatialDropout1D(embedding_dropout_rate, name="embedding_dropout"))

    # 2. LSTM Layers
    for i in range(num_lstm_layers):
        return_sequences = (
            i < num_lstm_layers - 1
        )  # True untuk semua kecuali layer LSTM terakhir

        lstm_layer = layers.LSTM(
            lstm_units,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_reg_strength),
            recurrent_regularizer=l2(l2_reg_strength),
            bias_regularizer=l2(l2_reg_strength),
            recurrent_dropout=recurrent_dropout_rate,  # Dropout di dalam sel LSTM
            name=f"lstm_layer_{i+1}",
        )

        if bidirectional:
            model.add(
                layers.Bidirectional(lstm_layer, name=f"bidirectional_lstm_{i+1}")
            )
        else:
            model.add(lstm_layer)

        # Dropout setelah setiap layer LSTM/Bidirectional LSTM
        model.add(layers.Dropout(dropout_rate, name=f"dropout_lstm_{i+1}"))

    # 3. Dense Output Layer
    model.add(
        layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=l2(l2_reg_strength),
            bias_regularizer=l2(l2_reg_strength),
            name="output_dense_layer",
        )
    )

    # Kompilasi model
    optimizer = Adam(
        learning_rate=learning_rate, clipnorm=1.0
    )  
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
