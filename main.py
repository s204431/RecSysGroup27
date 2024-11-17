import torch
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)

    # Hvis du bruger GPU (CUDA), så sæt seed for alle GPU-enheder
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For at sætte seed på alle CUDA enheder

    # For at få deterministiske resultater på CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slå off auto-tuning af CUDA-kernels for deterministisk output


class NewsEncoder(nn.Module):
    def __init__(self, e_dim, M, head_dim, batch_size):
        super(NewsEncoder, self).__init__()
        self.e_dim = e_dim
        self.M = M
        self.head_dim = head_dim
        self.batch_size = batch_size

        self.Q = nn.Parameter(torch.randn(size=(head_dim, e_dim, e_dim)))
        self.V = nn.Parameter(torch.randn(size=(head_dim, e_dim, e_dim)))
        self.q = nn.Parameter(torch.randn(size=(1, )))

    def forward(self, e_matrix):

        alpha = torch.zeros(size=(self.head_dim, e_matrix.shape[1], e_matrix.shape[1]))
        H = torch.zeros(size=(self.head_dim, e_matrix.shape[0], e_matrix.shape[1]))
        print(H.shape)

        for k in range(self.Q.shape[0]):

            Qe_m = torch.matmul(self.Q[k], e_matrix)
            for i in range(e_matrix.shape[1]):
                for j in range(e_matrix.shape[1]):

                    e_i = e_matrix[:, i].view(-1, 1)
                    e_j = e_matrix[:, j].view(-1, 1)

                    Qe_j = torch.matmul(self.Q[k], e_j)
                    e_iQe_j = torch.matmul(e_i.transpose(-2, -1), Qe_j)
                    e_iQe_m =  torch.matmul(e_i.transpose(-2, -1), Qe_m)

                    alpha[k,i,j] = torch.exp(e_iQe_j)/(torch.sum(torch.exp(e_iQe_m)))

        for k in range(self.Q.shape[0]):
            for i in range(e_matrix.shape[1]):
                a_ij_e_j = torch.sum(alpha[k, i, :] * e_matrix, dim=1)
                H[k, :, i] = torch.matmul(self.V[k], a_ij_e_j)

        H_reshaped = H.reshape(-1, e_matrix.shape[1])
        V_reshaped = self.V.reshape(-1, e_matrix.shape[0])

        a = torch.zeros(size=(1, e_matrix.shape[1]))
        
        for i in range(a.shape[1]):
            
                    
        return alpha


def main():
    set_seed(42)
    M =  2              # Number of tokens from the title
    batch_size = 1 
    e_dim = 3           # The dimension for the token 
    r_dim = 16          # The dimension of the output (r_dim, 1) 
    head_dim = 3        # The dimension of the header

    e_matrix = torch.randn((e_dim, M))
    M = NewsEncoder(e_dim, M, head_dim, batch_size)
    M.forward(e_matrix)
    #print('alpha\n')
    #print(M.forward(e_matrix))

































if __name__ == '__main__':
    main()