#pip install ipywidgets rich seaborn torch tokenizers sentencepiece sacremoses --quiet

import torch
from torch import nn
import math
from functools import partial
from pathlib import Path
from tqdm import tqdm
import rich
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tokenizers
import zipfile
from huggingface_hub import hf_hub_download
from NewsEmbedder import NewsEmbedder

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, d_model, d_model_out, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model_out % h == 0
        # We assume d_v always equals d_k
        self.d_k = torch.tensor(d_model_out // h)
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model_out) for _ in range(3)])
        self.linears.append(nn.Linear(d_model_out, d_model_out))
        self.attn = None # store the attention maps
        self.dropout = nn.Dropout(p=dropout)
        self.q = nn.Parameter(torch.randn(size=(d_model_out,)))

    """MULTI-HEADED SELF_ATTENTION"""
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        #d_k = torch.tensor(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -math.inf)
        p_attn = nn.functional.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def attention2(self, h_vectors):
        a = torch.matmul(torch.tanh(self.linears[-1](h_vectors)), self.q)
        #print(a.shape)
        alpha = torch.softmax(a, dim=-1)
        #print("Dimensions before einsum: ", alpha.shape, h_vectors.shape)
        return torch.einsum("bk,bkh->bh", alpha, h_vectors)  #Returns r
    
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        #print("Batches: ", nbatches)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        #print("1: ", query.shape, key.shape, value.shape)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        #print("2: ", query.shape, key.shape, value.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        #print("3: ", x.shape)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        #print("4: ", x.shape)
        #return self.linears[-1](x)
        return self.attention2(x)
