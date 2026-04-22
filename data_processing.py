import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

DATA_DIR = 'data'
ZIP_PATH = os.path.join(DATA_DIR, 'UCI_HAR_Dataset.zip')
EXTRACT_DIR = os.path.join(DATA_DIR, 'UCI HAR Dataset')
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading UCI HAR Dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete.")
        
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")

def load_signal(subset, signal):
    """
    Load a single signal from the text file.
    Returns a numpy array of shape (num_samples, 128).
    """
    filename = os.path.join(EXTRACT_DIR, subset, 'Inertial Signals', f"{signal}_{subset}.txt")
    df = pd.read_csv(filename, sep=r'\s+', header=None)
    return df.values

def load_signals(subset):
    """
    Load all 9 signals for a given subset (train or test).
    Returns a numpy array of shape (num_samples, 128, 9).
    """
    signals_data = []
    for signal in SIGNALS:
        signals_data.append(load_signal(subset, signal))
    # Stack along the features axis
    return np.dstack(signals_data)

def load_labels(subset):
    """
    Load labels. Labels are 1-6. We subtract 1 to make them 0-5.
    Returns a numpy array of shape (num_samples,).
    """
    filename = os.path.join(EXTRACT_DIR, subset, f"y_{subset}.txt")
    df = pd.read_csv(filename, sep=r'\s+', header=None)
    return df.values.flatten() - 1

def get_data():
    """
    Download, extract, load, and standardize the dataset.
    """
    download_and_extract()
    
    print("Loading training data...")
    X_train = load_signals('train')
    y_train = load_labels('train')
    
    print("Loading test data...")
    X_test = load_signals('test')
    y_test = load_labels('test')
    
    # Standardize data per feature
    # Calculate mean and std on training data
    num_samples_train, seq_len, num_features = X_train.shape
    
    # Flatten to calculate mean and std across all samples and timesteps
    X_train_flat = X_train.reshape(-1, num_features)
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)
    
    # Standardize
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    get_data()
