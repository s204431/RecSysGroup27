#pip install ipywidgets rich seaborn torch datasets transformers tokenizers sentencepiece sacremoses --quiet

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
import transformers
import tokenizers
import datasets
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

H, attention_map = attention(vectors, vectors, vectors)

# visualized the log of the attention map
tokens = [glove_vocabulary[x] for x in token_ids]
plot_attention_map(attention_map.log(), tokens, tokens, print_values=True)