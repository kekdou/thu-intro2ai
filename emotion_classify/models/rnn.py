import torch
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, hidden_size, num_layers, bidirectional, dropout):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding_dim = embedding_matrix.size(1)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embedded)
        if self.lstm.bidirectional:
            out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            out = h_n[-1, :, :]
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
