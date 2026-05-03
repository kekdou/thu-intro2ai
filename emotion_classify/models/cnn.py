import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, filter_sizes, num_filters, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.embedding_dim = embedding_matrix.size(1)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=num_filters, 
                      kernel_size=(k, self.embedding_dim))
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x).squeeze(3))
            pooled_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled_out)
        x_concat = torch.cat(pooled_outputs, dim=1)
        x_drop = self.dropout(x_concat)
        logits = self.fc(x_drop)
        return logits