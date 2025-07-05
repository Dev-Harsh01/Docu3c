import numpy as np
import pandas as pd
from scipy import stats
import os

def load_inertial_data(base_path, dataset='train'):
    """Load and preprocess inertial sensor data."""
    signals = ['body_acc_x', 'body_acc_y', 'body_acc_z', 
               'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    data = []
    
    for signal in signals:
        file_path = os.path.join(base_path, dataset, 'Inertial Signals', f'{signal}_{dataset}.txt')
        signal_data = np.loadtxt(file_path)
        data.append(signal_data)
    
    # Stack signals: shape (samples, timesteps, channels)
    data = np.stack(data, axis=2)  # (N, 128, 6)
    
    # Load subject IDs
    subject_path = os.path.join(base_path, dataset, f'subject_{dataset}.txt')
    subjects = np.loadtxt(subject_path, dtype=int)
    
    return data, subjects

def normalize_orientation(data):
    """Handle portrait/landscape by normalizing data across axes."""
    mag = np.sqrt(np.sum(data**2, axis=2, keepdims=True))
    normalized_data = data / (mag + 1e-10)  # Avoid division by zero
    return normalized_data

def preprocess_data(base_path, dataset='train'):
    """Preprocess inertial data for training/testing."""
    data, subjects = load_inertial_data(base_path, dataset)
    
    # Normalize orientation
    data = normalize_orientation(data)
    
    # Segment into windows (128 timesteps, no overlap for efficiency)
    return data, subjects

if __name__ == "__main__":
    base_path = "dataset"
    X_train, y_train = preprocess_data(base_path, 'train')
    X_test, y_test = preprocess_data(base_path, 'test')
    
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)