import torch
from torch import nn
from huggingface_hub import hf_hub_download
from NewsEmbedder import NewsEmbedder
from MHSA import MultiHeadedAttention

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NewsEncoder(nn.Module):
    """???."""
    def __init__(self, d_model_out, h=15, dropout=0.1):
        super(NewsEncoder, self).__init__()
        self.h = h
        self.dropout = dropout
        self.newsEmbedder = NewsEmbedder()
        self.MHSA = MultiHeadedAttention(h=self.h, d_model=300, d_model_out=d_model_out, dropout=self.dropout)
    
    def forward(self, token_ids):
        vectors = self.newsEmbedder(token_ids)
        return self.MHSA(vectors, vectors, vectors)