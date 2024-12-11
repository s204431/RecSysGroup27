import torch
from torch import nn
from NewsEmbedder import NewsEmbedder
from MHSA import MultiHeadedAttention

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NewsEncoder(nn.Module):
    """???."""
    def __init__(self, nlp, d_model_out, h=16, dropout=0.2):
        super(NewsEncoder, self).__init__()
        self.h = h
        self.dropout = dropout
        self.newsEmbedder = NewsEmbedder(nlp, dropout=self.dropout).to(DEVICE)
        self.nlp = nlp
        self.MHSA = MultiHeadedAttention(h=self.h, d_model=300, d_model_out=d_model_out, dropout=0.0).to(DEVICE)
    
    def forward(self, token_ids):
        mask = (token_ids != len(self.nlp.vocab.vectors)+1).float()
        mask = torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(1))
        vectors = self.newsEmbedder(token_ids)
        return self.MHSA(vectors, vectors, vectors, mask=mask)