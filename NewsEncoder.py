#! pip install ipywidgets rich seaborn torch datasets transformers tokenizers sentencepiece sacremoses --quiet

#%matplotlib inline

import torch
from torch import nn
import math
from functools import partial
from pathlib import Path
from tqdm import tqdm
import rich
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import transformers
import tokenizers
import zipfile
from huggingface_hub import hf_hub_download

#sns.set()   # Alters the style of plots to give them a more modern look.

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

def embed_sentence(sentence):
    """Tokenizes and embeds a sentence"""
    encoding = glove_tokenizer.encode(sentence, add_special_tokens=False)
    token_ids = encoding.ids
    vectors = [glove_vectors[t] for t in token_ids]
    return encoding, token_ids, vectors


glove_vocabulary, glove_vectors = load_glove_vectors()
glove_tokenizer = create_glove_tokenizer(glove_vocabulary)

# Tokenize and embed sentence
sentence = "This makes a little more sense now!"
encoding, token_ids, vectors = embed_sentence(sentence)

# Report (+ example of how to decode token ids)
rich.print(f"glove_vocabulary: type={type(glove_vocabulary)}, length={len(glove_vocabulary)}\n")
rich.print(f"glove_vectors: type={type(glove_vectors)}, shape={glove_vectors.shape}, dtype={glove_vectors.dtype}\n")
rich.print(f"Input sentence: [bold blue]`{sentence}`\n")
rich.print(f"sentence converted into {len(token_ids)} tokens (vocabulary: {glove_tokenizer.get_vocab_size()} tokens)\n")
rich.print(f"Tokens:\n{[glove_tokenizer.decode([t]) for t in token_ids]}\n")
rich.print(f"Token ids:\n{[t for t in token_ids]}\n")
for t,v in zip(token_ids, vectors):
    token_info = f"[blue]{glove_tokenizer.decode([t]):5}[/blue] (token id: {t:4})"
    vector_info = f"shape={v.shape}, mean={v.mean():.3f}, std={v.std():.3f}"
    rich.print(f" * {token_info} -> {vector_info}")
    #rich.print(f" * {token_info} -> {v}")

