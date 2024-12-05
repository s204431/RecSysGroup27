import torch
from torch import nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TimeEmbedder(nn.Module):
    def __init__(self, d_model_out):
        super(TimeEmbedder, self).__init__()
        self.w = nn.Parameter(torch.randn(size=(1,)))
        self.b = nn.Parameter(torch.randn(size=(1,)))
        #self.ws = nn.Parameter(torch.randn(size=(1,)))
        #self.wc = nn.Parameter(torch.randn(size=(1,)))
        #self.ws1 = nn.Parameter(torch.randn(size=(1,)))
        #self.bs = nn.Parameter(torch.randn(size=(1,)))
        #self.ws2 = nn.Parameter(torch.randn(size=(1,)))
        #self.bs2 = nn.Parameter(torch.randn(size=(1,)))
        #self.wc1 = nn.Parameter(torch.randn(size=(1,)))
        #self.bc = nn.Parameter(torch.randn(size=(1,)))
        #self.wc2 = nn.Parameter(torch.randn(size=(1,)))
        #self.bc2 = nn.Parameter(torch.randn(size=(1,)))
        self.ns = nn.Parameter(torch.randn(size=(1,)))
        self.nc = nn.Parameter(torch.randn(size=(1,)))
        self.d_model = d_model_out
        self.indices = torch.arange(start=0, end=self.d_model/2.0, step=1).to(DEVICE)
        self.linear = nn.Linear(d_model_out, d_model_out)
        #self.tanh = nn.Tanh()


    def forward(self, news_vectors, times):
        indices = self.indices.repeat(news_vectors.shape[0], 1)
        times_sin = torch.sin(times.unsqueeze(1)/torch.pow(torch.abs(self.ns*10000.0), 2*indices/self.d_model))
        times_cos = torch.cos(times.unsqueeze(1)/torch.pow(torch.abs(self.nc*10000.0), 2*indices/self.d_model))
        time_embedding = torch.stack((times_sin, times_cos), dim=2).reshape(times_sin.shape[0], -1)
        time_embedding = self.linear(time_embedding)
        return torch.cat((news_vectors, time_embedding), dim=-1)# + times_cos.unsqueeze(1)

#timeEmbedder = TimeEmbedder(4).to(DEVICE)
#print(timeEmbedder(torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]]).to(DEVICE), torch.tensor([1, 2]).to(DEVICE)))