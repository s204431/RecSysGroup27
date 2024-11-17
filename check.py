import torch
import torch.nn as nn
import numpy as np

if __name__ == '__main__':
    
    torch.manual_seed(32)
    head_dim = 2
    M = 3
    ########################################################################
    # Check for calculation of alpha
    ########################################################################

    # Create a 3x4 matrix with values from 1 to 12
    e_matrix = torch.randn(size = (3, 4))

    Q = nn.Parameter(torch.randn(size=(head_dim, M, M), dtype=torch.float))

    Qe = torch.matmul(Q, e_matrix)
    S = torch.matmul(e_matrix.transpose(-2, -1), Qe)
    alpha = torch.exp(S)/torch.sum(S, dim=-1, keepdim = True)

    alpha_loop = torch.zeros(size=(head_dim, e_matrix.shape[1], e_matrix.shape[1]))
    
    for k in range(Q.shape[0]):
        Qe_m = torch.matmul(Q[k], e_matrix)
        for i in range(e_matrix.shape[1]):
            for j in range(e_matrix.shape[1]):
                e_i = e_matrix[:, i].view(-1, 1)
                e_j = e_matrix[:, j].view(-1, 1)

                Qe_j = torch.matmul(Q[k], e_j)
                e_iQe_j = torch.matmul(e_i.transpose(-2, -1), Qe_j)
                e_iQe_m =  torch.matmul(e_i.transpose(-2, -1), Qe_m)
                print(e_i.transpose(-2, -1))
                print(Qe_m)
                print(e_iQe_m)
                print(torch.exp(e_iQe_m))
                print('i, j, k', i, j, k)
                print()
                print(torch.sum(torch.exp(e_iQe_m)))
                alpha_loop[k,i,j] = torch.exp(e_iQe_j)/(torch.sum(torch.exp(e_iQe_m)))

    print('\nalpha')
    print(alpha_loop)
    print('\ne_matrix')
    print(e_matrix)
    print('\nQ')
    print(Q)

    print('\n\n')
    print('Manuel check')
    print('I\'m going to check for i = 1, j = 2, k = 2\n')

    Q_2 = np.array([[ 0.6711, -1.3992,  0.3423],
                    [-0.1478,  1.7151, -0.0631],
                    [ 0.1238,  1.0855,  0.5981]])

    e_matrix = np.array([[ 0.8651,  0.0284,  0.5256, -0.3633],
                         [-0.4169, -1.2650,  1.2367,  0.1980],
                         [-1.5811,  0.4532,  2.3502, -0.3888]])
    
    
    

