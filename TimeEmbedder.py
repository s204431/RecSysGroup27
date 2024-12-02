import torch
from torch import nn

class TimeEmbedder(nn.Module):
    def __init__(self):
        super(TimeEmbedder, self).__init__()
        self.w = nn.Parameter(torch.randn(size=(1,)))
        self.b = nn.Parameter(torch.randn(size=(1,)))


    def forward(self, news_vectors, times):
        times = self.w*(times/100) + self.b
        return news_vectors + times.unsqueeze(1)