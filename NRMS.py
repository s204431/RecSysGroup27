import torch
from torch import nn
from MHSA import MultiHeadedAttention
from NewsEncoder import NewsEncoder

# define the device to use
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NRMS(nn.Module):
    """A simple Multi-head attention layer."""
    def __init__(self, nlp, h, dropout=0.2):
        "Take in model size and number of heads."
        super(NRMS, self).__init__()
        self.h = h
        self.dropout = dropout
        self.d_model_out = 256
        self.nlp = nlp
        self.news_encoder = NewsEncoder(nlp=nlp, d_model_out=self.d_model_out, h=self.h, dropout=self.dropout).to(DEVICE)
        self.MHSA = MultiHeadedAttention(h=self.h, d_model=self.d_model_out, d_model_out=self.d_model_out, dropout=0.0).to(DEVICE)

    def forward(self, history, targets):
        batch_size = history.shape[0]
        history_size = history.shape[1]
        targets_size = targets.shape[1]

        encoded_history = self.news_encoder(history.contiguous().view(-1, *history.shape[2:]))
        encoded_history = encoded_history.contiguous().view(batch_size, history_size, *encoded_history.shape[1:])
        mask = 1 - (history == len(self.nlp.vocab.vectors)+1).all(dim=-1).float()
        mask = torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(1))
        u = self.MHSA(encoded_history, encoded_history, encoded_history, mask=mask)

        r = self.news_encoder(targets.contiguous().view(-1, *targets.shape[2:]))
        r = r.contiguous().view(batch_size, targets_size, *r.shape[1:])
        dot_product = torch.bmm(r, u.unsqueeze(2)).squeeze(2)
        return nn.LogSoftmax(dim=1)(dot_product)
