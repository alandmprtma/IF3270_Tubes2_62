# CNN Model Creation Script
from tensorflow.keras import layers, models

def create_cnn_model(conv_layers=2, filters_per_layer=[32, 64], 
                     kernel_sizes=[3, 3], pooling_type='max'):
    """
    Create CNN model with configurable architecture
    
    Parameters:
    -----------
    conv_layers : int
        Number of convolutional layers
    filters_per_layer : list
        Number of filters for each convolutional layer
    kernel_sizes : list
        Kernel sizes for each convolutional layer
    pooling_type : str
        Type of pooling ('max' or 'average')
    
    Returns:
    --------
    tensorflow.keras.Model
        Compiled CNN model
    """
    model = models.Sequential()
    
    # Add convolutional layers
    for i in range(conv_layers):
        if i == 0:
            # First layer needs input shape
            model.add(layers.Conv2D(
                filters_per_layer[i], 
                (kernel_sizes[i], kernel_sizes[i]), 
                activation='relu', 
                input_shape=(32, 32, 3),
                padding='valid',
                name=f'conv2d_{i+1}'
            ))
            print(f"After Conv Layer {i + 1}, output shape: {model.output_shape}")
        else:
            model.add(layers.Conv2D(
                filters_per_layer[i], 
                (kernel_sizes[i], kernel_sizes[i]), 
                activation='relu',
                padding='valid',
                name=f'conv2d_{i+1}'
            ))
            print(f"After Conv Layer {i + 1}, output shape: {model.output_shape}")
        
        # Add pooling layer after each conv layer
        if pooling_type == 'max':
            model.add(layers.MaxPooling2D((2, 2), name=f'max_pooling2d_{i+1}'))
        else:
            model.add(layers.AveragePooling2D((2, 2), name=f'average_pooling2d_{i+1}'))
    
    # Flatten layer
    model.add(layers.Flatten(name='flatten'))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dense(10, activation='softmax', name='output_dense'))
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
