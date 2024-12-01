import torch
from torch import nn
from huggingface_hub import hf_hub_download
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
        self.MHSA = MultiHeadedAttention(h=self.h, d_model=300, d_model_out=d_model_out, dropout=0.0).to(DEVICE)
    
    def forward(self, token_ids):
        vectors = self.newsEmbedder(token_ids)
        return self.MHSA(vectors, vectors, vectors)