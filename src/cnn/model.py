import config
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def create_cnn_model(
    input_shape=(224, 224, 3),
    num_conv_layers=config.NUM_CONV_LAYERS,
    base_filters=config.BASE_FILTERS,
    kernel_size=config.KERNEL_SIZE,
    pool_size=config.POOL_SIZE,
    dense_units=config.DENSE_UNITS,
    num_dense_layers=config.NUM_DENSE_LAYERS,
    dropout_rate=config.DROPOUT_RATE,
    l2_reg=config.L2_REG,
    num_classes=3,
    learning_rate=config.LEARNING_RATE,
):
    """
    Membuat model CNN untuk klasifikasi gambar
    
    Parameters:
    -----------
    input_shape : tuple
        Bentuk input gambar (height, width, channels)
    num_conv_layers : int
        Jumlah blok konvolusi
    base_filters : int
        Jumlah filter awal, akan meningkat di setiap layer
    kernel_size : tuple
        Ukuran kernel konvolusi
    pool_size : tuple
        Ukuran pooling window
    dense_units : int
        Jumlah unit di dense layer
    num_dense_layers : int
        Jumlah dense layer sebelum output
    dropout_rate : float
        Rate untuk dropout layer
    l2_reg : float
        Koefisien regularisasi L2
    num_classes : int
        Jumlah kelas untuk klasifikasi
    learning_rate : float
        Learning rate untuk optimizer
    
    Returns:
    --------
    tensorflow.keras.Model
        Model CNN yang sudah dikompile
    """
    
    model = models.Sequential(name="CNN_Classifier")
    
    # Input layer
    model.add(layers.InputLayer(input_shape=input_shape, name="input_layer"))
    
    # Blok konvolusi dengan pooling
    current_filters = base_filters
    
    for i in range(num_conv_layers):
        # Layer Konvolusi dengan regularisasi L2
        model.add(
            layers.Conv2D(
                filters=current_filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(l2_reg),
                bias_regularizer=l2(l2_reg),
                name=f"conv2d_layer_{i+1}"
            )
        )
        
        # Batch Normalization untuk stabilitas training
        model.add(
            layers.BatchNormalization(name=f"batch_norm_{i+1}")
        )
        
        # Max Pooling untuk mengurangi dimensi spasial
        model.add(
            layers.MaxPooling2D(
                pool_size=pool_size,
                strides=(2, 2),
                padding='valid',
                name=f"max_pooling_{i+1}"
            )
        )
        
        # Dropout untuk mencegah overfitting
        model.add(
            layers.Dropout(dropout_rate, name=f"dropout_conv_{i+1}")
        )
        
        # Tingkatkan jumlah filter untuk layer berikutnya
        current_filters = min(current_filters * 2, 512)  # Cap maksimal 512 filter
    
    # Flatten untuk transisi ke dense layers
    model.add(layers.Flatten(name="flatten_layer"))
    
    # Dense layers dengan dropout
    for i in range(num_dense_layers):
        # Kurangi jumlah unit secara bertahap
        units = dense_units // (2 ** i) if i > 0 else dense_units
        units = max(units, 64)  # Minimal 64 unit
        
        model.add(
            layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l2(l2_reg),
                bias_regularizer=l2(l2_reg),
                name=f"dense_layer_{i+1}"
            )
        )
        
        # Batch Normalization
        model.add(
            layers.BatchNormalization(name=f"dense_batch_norm_{i+1}")
        )
        
        # Dropout
        model.add(
            layers.Dropout(dropout_rate, name=f"dropout_dense_{i+1}")
        )
    
    # Layer output dengan softmax untuk klasifikasi multiclass
    model.add(
        layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=l2(l2_reg),
            bias_regularizer=l2(l2_reg),
            name="output_dense_layer"
        )
    )
    

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"],
    )
    
    return model