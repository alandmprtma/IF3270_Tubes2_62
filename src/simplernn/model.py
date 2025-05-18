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
    """
    Create a balanced RNN model for text classification on small datasets.
    All default values are pulled from config.py.
    """

    model = models.Sequential(name="Simple_RNN_Classifier")

    # Add embedding layer with regularization
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embedding_layer",
        )
    )

    # Add embedding dropout
    model.add(layers.SpatialDropout1D(config.EMBEDDING_DROPOUT, name="embedding_dropout"))

    # Add RNN layer(s)
    for i in range(num_rnn_layers):
        return_sequences = i < num_rnn_layers - 1

        rnn_layer = layers.SimpleRNN(
            rnn_units,
            return_sequences=return_sequences,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            recurrent_dropout=config.RECURRENT_DROPOUT,
            name=f"simplernn_layer_{i+1}",
        )

        if bidirectional:
            model.add(layers.Bidirectional(rnn_layer, name=f"bidirectional_rnn_{i+1}"))
        else:
            model.add(rnn_layer)

        # Add dropout after each RNN layer
        model.add(layers.Dropout(dropout_rate, name=f"dropout_rnn_{i+1}"))

    # Add final dense layer for classification
    model.add(
        layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            name="output_dense_layer",
        )
    )

    # Compile model with gradient clipping
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model