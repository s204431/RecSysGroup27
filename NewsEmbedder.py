import torch
from torch import nn

class NewsEmbedder(nn.Module):
    def __init__(self, nlp, dropout=0.2):
        super(NewsEmbedder, self).__init__()
        self.embeddingDimension = 300
        data = (nlp.vocab.vectors.data/10.0).tolist()
        data.append([0.0 for _ in range(0, self.embeddingDimension)])
        data.append([0.0 for _ in range(0, self.embeddingDimension)])
        vectors = torch.tensor(data)
        self.embeddings = torch.nn.Embedding(vectors.shape[0], vectors.shape[1], padding_idx=vectors.shape[0]-1)
        self.embeddings.weight.data = vectors
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids):
        vectors = self.embeddings(token_ids)
        if self.dropout is not None:
            vectors = self.dropout(vectors)
        return vectors