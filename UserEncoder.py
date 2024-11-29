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
        encoded_history = torch.stack([self.news_encoder(title).squeeze() for title in history], 0).unsqueeze(0)
        #print("Encoded history shape", encoded_history.shape)
        u = self.MHSA(encoded_history, encoded_history, encoded_history)

        predictions = []
        for i in range(len(targets)):
            r = self.news_encoder(targets[i])
            predictions.append(torch.dot(u.squeeze(), r.squeeze()))
        return torch.stack(predictions)


#history = ["This a test", "This is the click history"]
#target = ["I hope this works!", "with multiple targets"]
#user_encoder = UserEncoder(h=16, dropout=0.2)
#output = user_encoder(history=history, targets=target)

#rich.print(output.shape)
#rich.print(output)
