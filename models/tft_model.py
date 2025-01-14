import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, num_layers=2, dropout_rate=0.2):
        super(TemporalFusionTransformer, self).__init__()

        # Static variable input embedding
        self.static_embedding = nn.Linear(input_size, hidden_size)

        # Temporal variable embedding
        self.temporal_embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = nn.Embedding(1000, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Fully connected layers for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, static_data=None):
        """
        Args:
            x: (batch_size, sequence_length, input_size) - Temporal input features
            static_data: (batch_size, input_size) - Static input features
        Returns:
            output: (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # Temporal embedding
        temporal_encoded = self.temporal_embedding(x)

        # Add positional encoding
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(x.device)
        positional_encoded = self.positional_encoding(positions)
        x = temporal_encoded + positional_encoded

        # Static embedding (if provided)
        if static_data is not None:
            static_encoded = self.static_embedding(static_data).unsqueeze(1)  # Add time dimension
            x = x + static_encoded

        # Pass through transformer encoder
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)

        # Use only the last time step
        x = x[:, -1, :]

        # Fully connected layer
        output = self.fc(self.dropout(x))

        return output


# Example instantiation for debugging
if __name__ == "__main__":
    batch_size = 32
    sequence_length = 96
    input_size = 8
    hidden_size = 64
    output_size = 96
    num_heads = 4
    num_layers = 2
    dropout_rate = 0.2

    model = TemporalFusionTransformer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    # Example data
    temporal_data = torch.rand(batch_size, sequence_length, input_size)
    static_data = torch.rand(batch_size, input_size)

    # Forward pass
    output = model(temporal_data, static_data)
    print("Output shape:", output.shape)
