import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model by matching the remaining elements
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class HARTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=3, num_classes=6, dim_feedforward=128, dropout=0.1):
        super(HARTransformer, self).__init__()
        
        # Project input features to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Project input
        x = self.input_projection(x) # (batch_size, seq_len, d_model)
        
        # PyTorch transformer expects input shape (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x) # (seq_len, batch_size, d_model)
        
        # For sequence classification, we can take the mean over the sequence length, 
        # or take the output corresponding to the last timestep. Let's use mean pooling.
        x = x.mean(dim=0) # (batch_size, d_model)
        
        # Classify
        logits = self.classifier(x) # (batch_size, num_classes)
        
        return logits

if __name__ == '__main__':
    # Test model with dummy data
    batch_size = 32
    seq_len = 128
    input_dim = 9
    
    model = HARTransformer(input_dim=input_dim)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
