import torch.nn as nn
import torch
import math
from torch import Tensor
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, num_heads: int, num_layers: int, dropout: float = 0.1,
                 device='cuda'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, emb_size)  # Add 1 for the classification token
        self.positional_encoding = PositionalEncoding(emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.vocab_cardinality = vocab_size
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``
        """
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        # Transformer encoding
        x = self.transformer_encoder(x)

        return torch.mean(x, dim=0) # Return the average direction

class PHOCEncoder(nn.Module):
    def __init__(self, phoc_vector_size, node_size, device='cuda'):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(phoc_vector_size, node_size),
            nn.ReLU(),
            nn.Linear(node_size, node_size)
        )
        self.device=device
        self.to(device)
    def forward(self, batch):
        return self.projection(batch.to(self.device))