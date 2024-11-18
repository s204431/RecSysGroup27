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
from MHSA import MultiHeadedAttention
from NewsEncoder import NewsEncoder

sns.set()

# define the device to use
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class UserEncoder(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, dropout=0.2):
        "Take in model size and number of heads."
        super(UserEncoder, self).__init__()
        self.h = h
        self.dropout = dropout
        self.d_model_out = 208
        self.news_encoder = NewsEncoder(d_model_out=self.d_model_out, h=self.h, dropout=self.dropout)
        self.MHSA = MultiHeadedAttention(h=self.h, d_model=self.d_model_out, d_model_out=self.d_model_out, dropout=0.0)

    def forward(self, history, target):
        encoded_history = torch.stack([self.news_encoder(title).squeeze() for title in history], 0).unsqueeze(0)
        r = self.news_encoder(target)
        print("Encoded history shape", encoded_history.shape)
        u = self.MHSA(encoded_history, encoded_history, encoded_history)
        return torch.dot(u.squeeze(), r.squeeze())

history = ["This a test", "This is the click history"]
target = "I hope this works!"
user_encoder = UserEncoder(h=16, dropout=0.2)
output = user_encoder(history=history, target=target)

rich.print(output.shape)
rich.print(output)
