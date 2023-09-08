import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """
    Train the attribute inference model.

    Parameters:
    - model: tensorflow.keras.Model
      The deep learning model to train.
    - X_train: numpy.ndarray
      The training feature data.
    - y_train: numpy.ndarray
      The training target data.
    - X_test: numpy.ndarray
      The testing feature data.
    - y_test: numpy.ndarray
      The testing target data.
    - epochs: int
      The number of training epochs (default: 10).
    - batch_size: int
      The batch size for training (default: 32).

    Returns:
    - history: tensorflow.python.keras.callbacks.History
      Training history.
    """
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test))

    return history
def build_attribute_inference_model(input_shape, num_classes):
    """
    Build and return a deep learning model for attribute inference.

    Parameters:
    - input_shape: Tuple
      The shape of the input data.
    - num_classes: int
      The number of classes for attribute inference.

    Returns:
    - model: tensorflow.keras.Model
      The constructed deep learning model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
