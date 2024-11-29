#pip install ipywidgets rich seaborn torch tokenizers sentencepiece sacremoses --quiet

import torch
from torch import nn
import rich
import seaborn as sns
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
        self.d_model_out = 256
        self.news_encoder = NewsEncoder(d_model_out=self.d_model_out, h=self.h, dropout=self.dropout).to(DEVICE)
        self.MHSA = MultiHeadedAttention(h=self.h, d_model=self.d_model_out, d_model_out=self.d_model_out, dropout=0.0).to(DEVICE)

    def forward(self, history, targets):
        batch_size = history.shape[0]
        history_size = history.shape[1]
        targets_size = targets.shape[1]

        encoded_history = self.news_encoder(history.contiguous().view(-1, *history.shape[2:]))
        encoded_history = encoded_history.contiguous().view(batch_size, history_size, *encoded_history.shape[1:])
        u = self.MHSA(encoded_history, encoded_history, encoded_history)

        r = self.news_encoder(targets.contiguous().view(-1, *targets.shape[2:]))
        r = r.contiguous().view(batch_size, targets_size, *r.shape[1:])
        dot_product = torch.bmm(r, u.unsqueeze(2)).squeeze(2)
        return nn.LogSoftmax(dim=1)(dot_product)


#history = ["This a test", "This is the click history"]
#target = ["I hope this works!", "with multiple targets"]
#user_encoder = UserEncoder(h=16, dropout=0.2)
#output = user_encoder(history=history, targets=target)

#rich.print(output.shape)
#rich.print(output)

#torch.manual_seed(42)
#history_input = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
#targets_input = torch.tensor([[[0, 1, 2], [3, 4, 5], [20000, 20000, 20000]], [[6, 7, 8], [9, 10, 11], [20000, 20000, 20000]]])
#user_encoder = UserEncoder(16, 0.2)
#output = user_encoder(history_input, targets_input)
#print(output)
#u = torch.tensor([[0, 1, 2], [3, 4, 5]])
#r = torch.tensor([[6, 7, 8], [9, 10, 11]])
#print(torch.sum(u * r, dim=1))


#news_encoder = NewsEncoder(d_model_out=16, h=16, dropout=0.2)
#input = torch.tensor([[0, 1, 2, 20000], [3, 4, 5, 20000]])
#print(news_encoder(input))