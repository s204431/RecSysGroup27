import torch
from torch import nn
from huggingface_hub import hf_hub_download
from NewsEmbedder import NewsEmbedder
from MHSA import MultiHeadedAttention

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NewsEncoder(nn.Module):
    """???."""
    def __init__(self, h=15, dropout=0.1):
        super(NewsEncoder, self).__init__()
        self.h = h
        self.dropout = dropout
        self.newsEmbedder = NewsEmbedder()
        self.MHSA = MultiHeadedAttention(h=self.h, d_model=self.newsEmbedder.embeddings.embedding_dim, dropout=self.dropout)
    
    def forward(self, string):
        vectors = self.newsEmbedder(string)
        return self.MHSA(vectors, vectors, vectors)

    
ne = NewsEncoder()
output = ne("My name is. My name is!")
print(output.shape)