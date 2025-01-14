import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def load_data(train_path, test_path, input_window=96, output_window=96):
    """
    Load and preprocess the data.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(train_data[['cnt']])
    y_test = scaler.transform(test_data[['cnt']])

    def create_sequences(data, target, input_window, output_window):
        X, y = [], []
        for i in range(len(data) - input_window - output_window):
            X.append(data[i:i + input_window, :])
            y.append(target[i + input_window:i + input_window + output_window])
        return np.array(X), np.array(y)

    feature_cols = [col for col in train_data.columns if col not in ['instant', 'dteday', 'casual', 'registered', 'cnt']]
    train_features = train_data[feature_cols].values
    test_features = test_data[feature_cols].values

    X_train, y_train = create_sequences(train_features, y_train, input_window, output_window)
    X_test, y_test = create_sequences(test_features, y_test, input_window, output_window)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.squeeze(-1), dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.squeeze(-1), dtype=torch.float32)

    return X_train, y_train, X_test, y_test, scaler