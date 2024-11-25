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
import random
from transformers import AutoTokenizer, AutoModel

class NewsEmbedder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__loadBertModel__()

    def __loadBertModel__(self, model_name="Maltehb/danish-bert-botxo"):
        """Loads a word2ved model from BERT"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)  # Freeze the model, so that it cannot be trained.
        self.embeddingDimension = self.model.config.hidden_size

    def forward(self, string):
        """Tokenizes and embeds a string"""
        input = self.tokenizer("Virker det eller hvad?", return_tensors="pt")
        output = self.model(**input)
        return output.last_hidden_state
        

'''
        # Load the GloVe vectors
        self.glove_vocabulary, self.glove_vectors = self.__load_glove_vectors()
        self.glove_tokenizer = self.__create_glove_tokenizer()

        # Instantiate enbeddings layer using GloVe vectors
        self.embeddings = nn.Embedding(*self.glove_vectors.shape)
        self.embeddings.weight.data = self.glove_vectors
        self.embeddings.weight.requires_grad = False    # Freeze embedding layer to disable backtracking (training)

    #(glove.6B.300d.txt, glove.6B.zip), (glove.42B.300d.txt, glove.42B.300d.zip), (glove.840B.300d.txt, glove.840B.300d.zip)
    def __load_glove_vectors(self, filename = "glove.6B.300d.txt") -> Tuple[List[str], torch.Tensor]:
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

        vocabulary.append("[UNK]")
        vectors.append(torch.tensor([random.uniform(-1, 1) for i in range(0, 300)]))
        vectors = torch.stack(vectors)
        return vocabulary, vectors
    
    def __create_glove_tokenizer(self):
        """Create tokenizer using GloVe vocabulary"""
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab={v:i for i,v in enumerate(self.glove_vocabulary)}, unk_token="[UNK]"))
        tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=False)
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        return tokenizer

    def forward(self, string):
        """Tokenizes and embeds a string"""
        encoding = self.glove_tokenizer.encode(string, add_special_tokens=True)
        token_ids = torch.tensor(encoding.ids)
        vectors = self.embeddings(token_ids)
        return vectors.unsqueeze(0)
'''