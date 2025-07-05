import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes=30):
    """Build 1D-CNN for gait authentication with 30 classes."""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Train and save the gait authentication model."""
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    
    # Adjust labels to 0-based indexing (1 to 30 -> 0 to 29)
    y_train = y_train - 1
    num_classes = 30  # Fixed to match UCI HAR dataset (30 subjects)
    
    model = build_cnn_model(input_shape=(128, 6), num_classes=num_classes)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    model.save('gait_cnn_model.h5')

if __name__ == "__main__":
    train_model()