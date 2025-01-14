import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.data_loader import load_data
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.tft_model import TemporalFusionTransformer
from models.autoformer_model import Autoformer

def train_model(model_name, train_path, test_path, input_window=96, output_window=96, hidden_size=64, num_layers=2,
                dropout_rate=0.2, batch_size=32, epochs=50, learning_rate=0.001, num_heads=4, dim_feedforward=128):
    """
    Train the specified model and evaluate with MSE and MAE over 5 runs.
    Save predictions and ground truths for plotting.
    """
    mse_scores = []
    mae_scores = []
    final_predictions = []
    final_ground_truths = []

    for run in range(5):
        X_train, y_train, X_test, y_test, scaler = load_data(train_path, test_path, input_window, output_window)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = X_train.shape[2]  # Number of features

        if model_name == 'LSTM':
            model = LSTMModel(input_size, hidden_size, num_layers, output_window, dropout_rate)
        elif model_name == 'Transformer':
            model = TransformerModel(input_size, output_window, num_heads, num_layers, dim_feedforward, dropout_rate)
        elif model_name == 'TFT':
            model = TemporalFusionTransformer(
                input_size=X_train.shape[2], 
                hidden_size=hidden_size,
                output_size=output_window,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
        elif model_name == 'Autoformer':
            model = Autoformer(
                input_size=input_size,
                output_size=output_window,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion_mse(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            print(f"Run {run + 1}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        mse_loss = 0.0
        mae_loss = 0.0
        run_predictions = []
        run_ground_truths = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                mse_loss += criterion_mse(outputs, y_batch).item()
                mae_loss += criterion_mae(outputs, y_batch).item()
                run_predictions.append(outputs.cpu().numpy())
                run_ground_truths.append(y_batch.cpu().numpy())

        mse_scores.append(mse_loss / len(test_loader))
        mae_scores.append(mae_loss / len(test_loader))

        final_predictions.append(np.concatenate(run_predictions, axis=0))
        final_ground_truths.append(np.concatenate(run_ground_truths, axis=0))

        print(f"Run {run + 1}, Test MSE: {mse_loss / len(test_loader):.4f}, Test MAE: {mae_loss / len(test_loader):.4f}")

    # Average predictions and ground truths across all runs
    final_predictions = np.mean(final_predictions, axis=0)
    final_ground_truths = np.mean(final_ground_truths, axis=0)

    # Save combined predictions and ground truths
    np.save(f"{model_name.lower()}_final_predictions_output{output_window}.npy", final_predictions)
    np.save(f"{model_name.lower()}_final_ground_truths_output{output_window}.npy", final_ground_truths)

    print(f"Final Results for Output Window {output_window}:")
    print(f"MSE - Mean: {np.mean(mse_scores):.4f}, Std: {np.std(mse_scores):.4f}")
    print(f"MAE - Mean: {np.mean(mae_scores):.4f}, Std: {np.std(mae_scores):.4f}")