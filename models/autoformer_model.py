import torch
import torch.nn as nn

class Autoformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_heads=4, num_layers=2, dropout_rate=0.2):
        super(Autoformer, self).__init__()

        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # Decomposition layer for trend and seasonal components
        self.decomposition = DecompositionLayer(kernel_size=25)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        batch_size, seq_len, num_features = x.size()
        assert num_features == self.input_embedding.in_features, \
            f"Input features ({num_features}) do not match expected features ({self.input_embedding.in_features})."

        # Input embedding
        x = self.input_embedding(x)

        # Decomposition into trend and seasonal components
        trend, seasonal = self.decomposition(x)

        # Use only seasonal components for the transformer encoder
        seasonal = seasonal.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        seasonal_encoded = self.encoder(seasonal)
        seasonal_encoded = seasonal_encoded.permute(1, 0, 2)  # Back to (batch_size, seq_len, hidden_size)

        # Aggregate trend and seasonal components
        combined = trend + seasonal_encoded

        # Use the last time step
        output = combined[:, -1, :]

        # Fully connected output
        output = self.fc(self.dropout(output))

        return output


class DecompositionLayer(nn.Module):
    def __init__(self, kernel_size):
        super(DecompositionLayer, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        """
        Decompose input into trend and seasonal components.
        Args:
            x: (batch_size, sequence_length, hidden_size)
        Returns:
            trend: (batch_size, sequence_length, hidden_size)
            seasonal: (batch_size, sequence_length, hidden_size)
        """
        x = x.permute(0, 2, 1)  # (batch_size, hidden_size, sequence_length)
        trend = self.moving_avg(x)  # Apply moving average
        trend = trend.permute(0, 2, 1)  # (batch_size, sequence_length, hidden_size)
        seasonal = x.permute(0, 2, 1) - trend
        return trend, seasonal
