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

sns.set()

# control verbosity (we can turn these on if we get too many logs at some point. Don't know if this will be relevant)
#transformers.logging.set_verbosity_error()
#datasets.logging.set_verbosity_error()

# define the device to use
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# define support functions
def load_glove_vectors(filename = "glove.6B.300d.txt") -> Tuple[List[str], torch.Tensor]:
    """Load the GloVe vectors. See: `https://github.com/stanfordnlp/GloVe`"""
    path = Path(hf_hub_download(repo_id="stanfordnlp/glove", filename="glove.6B.zip"))
    target_file = path.parent / filename
    if not target_file.exists():
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(path.parent)

    # parse the vocabulary and the vectors
    vocabulary = []
    vectors = []
    with open(target_file, "r", encoding="utf8") as f:
        for l in tqdm(f.readlines(), desc=f"Parsing {target_file.name}..." ):
            word, *vector = l.split()
            vocabulary.append(word)
            vectors.append(torch.tensor([float(v) for v in vector]))

    vectors = torch.stack(vectors)
    return vocabulary, vectors

def create_glove_tokenizer(vocabulary):
    """Create tokenizer using GloVe vocabulary"""
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab={v:i for i,v in enumerate(glove_vocabulary)}, unk_token="<|unknown|>"))
    tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=False)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return tokenizer

def embed_sentence(sentence, tokenizer, embeddings):
    """Tokenizes and embeds a sentence"""
    encoding = tokenizer.encode(sentence, add_special_tokens=False)
    token_ids = torch.tensor(encoding.ids)
    vectors = embeddings(token_ids)
    return encoding, token_ids, vectors


glove_vocabulary, glove_vectors = load_glove_vectors()
glove_tokenizer = create_glove_tokenizer(glove_vocabulary)

# Instantiate enbeddings layer using GloVe vectors
embeddings = nn.Embedding(*glove_vectors.shape)
embeddings.weight.data = glove_vectors
embeddings.weight.requires_grad = False     # Freeze if you don't want to train further (they do this in week 5)

# Tokenize and embed sentence
sentence = "This makes a little more sense now!"
encoding, token_ids, vectors = embed_sentence(sentence, glove_tokenizer, embeddings)
vectors = vectors.unsqueeze(0)

print(vectors.shape)
# Report
rich.print(f"glove_vocabulary: type={type(glove_vocabulary)}, length={len(glove_vocabulary)}\n")
rich.print(f"glove_vectors: type={type(glove_vectors)}, shape={glove_vectors.shape}, dtype={glove_vectors.dtype}\n")
rich.print(f"Input sentence: [bold blue]`{sentence}`\n")
rich.print(f"sentence converted into {len(token_ids)} tokens (vocabulary: {glove_tokenizer.get_vocab_size()} tokens)\n")
rich.print(f"Tokens:\n{[glove_tokenizer.decode([t]) for t in token_ids]}\n")
rich.print(f"Token ids:\n{[t for t in token_ids.numpy()]}\n")
for t,v in zip(token_ids, vectors):
    token_info = f"[blue]{glove_tokenizer.decode([t]):5}[/blue] (token id: {t:4})"
    vector_info = f"shape={v.shape}, mean={v.mean():.3f}, std={v.std():.3f}"
    rich.print(f" * {token_info} -> {vector_info}")
    #rich.print(f" * {token_info} -> {v}")



"""ATTEMPT TO PLOT ATTENTION BETWEEN EACH OF THE WORDS IN THE SENTENCE"""
def attention(Q, K, V, tau=None):
    """A simple parallelized attention layer"""
    if tau is None:
        tau = math.sqrt(float(Q.shape[-1]))
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[0] == V.shape[0]
    attention_map = Q @ K.T / tau
    attention_weights = attention_map.softmax(dim=1)
    return torch.einsum("qk, kh -> qh", attention_weights, V), attention_weights

def plot_attention_map(attention_map, queries_labels, keys_labels, print_values:bool=False, ax=None, color_bar:bool=True):
    """Plot the attention weights as a 2D heatmap"""
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6), dpi=150)
    else:
        fig = plt.gcf()
    im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
    ax.grid(False)
    ax.set_ylabel("$\mathbf{Q}$")
    ax.set_xlabel("$\mathbf{K}$")
    ax.set_yticks(np.arange(len(queries_labels)))
    ax.set_yticklabels(queries_labels)
    ax.set_xticks(np.arange(len(keys_labels)))
    ax.set_xticklabels(keys_labels)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if print_values:
        for i in range(len(queries_labels)):
            for j in range(len(keys_labels)):
                text = ax.text(j, i, f"{attention_map[i, j]:.2f}",
                            ha="center", va="center", color="w")

    if color_bar:
      fig.colorbar(im, fraction=0.02, pad=0.04)

    plt.subplots_adjust(left=0.1, right=0.9)
    fig.tight_layout()
    plt.show()

#H, attention_map = attention(vectors, vectors, vectors)

# visualized the log of the attention map
#tokens = [glove_vocabulary[x] for x in token_ids]
#plot_attention_map(attention_map.log(), tokens, tokens, print_values=True)

class MultiHeadedAttention(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None # store the attention maps
        self.dropout = nn.Dropout(p=dropout)
        self.q = nn.Parameter(torch.randn(size=(d_model,)))

    """MULTI-HEADED SELF_ATTENTION"""
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -math.inf)
        p_attn = nn.functional.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def attention2(self, h_vectors):
        a = torch.matmul(torch.tanh(self.linears[-1](h_vectors)), self.q)
        print(a.shape)
        alpha = torch.softmax(a, dim=-1)
        print("Dimensions before einsum: ", alpha.shape, h_vectors.shape)
        return torch.einsum("bk,bkh->bh", alpha, h_vectors)  #Returns r
    
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        print("Batches: ", nbatches)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        print("1: ", query.shape, key.shape, value.shape)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        print("2: ", query.shape, key.shape, value.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        print("3: ", x.shape)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        print("4: ", x.shape)
        #return self.linears[-1](x)
        return self.attention2(x)


MHSA = MultiHeadedAttention(h=10, d_model=embeddings.embedding_dim, dropout=0.1)
output = MHSA(vectors, vectors, vectors)

rich.print(output.shape)
rich.print(output)
#rich.print(MHSA.attn.shape)
#rich.print(MHSA.attn)
