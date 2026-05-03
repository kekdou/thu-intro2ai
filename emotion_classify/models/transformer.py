import torch
import torch.nn as nn
import math

class PosEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super(PosEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class Transformer(nn.Module):
    def __init__(self, embedding_matrix, num_classes, n_heads, n_layers, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding_dim = embedding_matrix.size(1)
        self.pos_encoder = PosEncoding(self.embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, 
            nhead=n_heads, 
            dim_feedforward=self.embedding_dim * 4, 
            dropout=dropout,
            batch_first=True, 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded)
        output = torch.mean(output, dim=1)
        logits = self.fc(output)
        return logits