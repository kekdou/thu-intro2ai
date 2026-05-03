import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embedding_matrix, num_classes, hidden_dims, dropout):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        embedding_dim = embedding_matrix.size(1)
        layers = []
        input_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        hidden_out = self.hidden_layers(pooled)
        logits = self.fc_out(hidden_out)
        return logits
