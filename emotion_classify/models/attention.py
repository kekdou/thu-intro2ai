import torch
import torch.nn as nn
import torch.nn.functional as F

class TextAttnRNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, hidden_size, num_layers, dropout):
        super(TextAttnRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding_dim = embedding_matrix.size(1)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn_weight = nn.Parameter(torch.randn(hidden_size * 2, 1))
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        M = torch.tanh(out)
        score = torch.matmul(M, self.attn_weight)
        alpha = F.softmax(score, dim=1)
        out_attn = torch.sum(out * alpha, dim=1)
        out_attn = self.dropout(out_attn)
        logits = self.fc(out_attn)
        return logits