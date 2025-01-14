import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads=4, num_layers=2, dim_feedforward=128, dropout_rate=0.2):
        """
        Transformer model definition.
        """
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_size, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(dim_feedforward, output_size)

    def forward(self, x):
        x = self.input_embedding(x)
        x = x.permute(1, 0, 2)  # Required shape for transformer: (sequence_length, batch_size, feature_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, sequence_length, feature_dim)
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.fc(x)
        return x