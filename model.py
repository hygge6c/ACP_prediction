import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout, max_len):
        super().__init__()
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_embed = self.embedding(x) * (self.embed_dim ** 0.5)
        x_embed = self.pos_encoder(x_embed.transpose(0, 1)).transpose(0, 1)
        padding_mask = (x == 0)
        x_encoded = self.encoder(x_embed, src_key_padding_mask=padding_mask)
        x_pooled = self.pooling(x_encoded.transpose(1, 2)).squeeze(-1)
        x_features = F.relu(self.fc1(x_pooled))
        x_features = self.dropout(x_features)
        out = self.fc2(x_features)
        return self.sigmoid(out).squeeze()