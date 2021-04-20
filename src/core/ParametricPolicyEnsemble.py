import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer
import numpy as np
from ParametricPolicy import ParametricPolicy
from random import choices
import ipdb
class ParametricPolicyEnsemble(nn.Module):
    
    def __init__(self, net_count, x_dim, u_dim, u_max, u_min,layers, hidden_size = 40, lr = 5e-4):
        
        super(ParametricPolicyEnsemble, self).__init__()

        self.policies = []
        for i in range(net_count):
            self.policies.append(ParametricPolicy(x_dim, u_dim, u_min, u_max, layers, hidden_size, lr))
    def bootstrap(self,x, u):
        N = x.shape[0]
        idx = np.arange(N).tolist()
        data_x = []
        data_u = []
        for i in range(len(self.policies)):
            idx_net = choices(idx, k = N)
            data_x.append(x[idx_net,:]) 
            data_u.append(u[idx_net,:]) 

        return data_x, data_u
    def train(self, states,actions, epochs = 200, batch_size = 256, split_ratio = 0.1):
        '''
            Need to deal with the validation. Bootstrapping creates duplicates and may
            cause low validation error ...
        '''

        data_x, data_u = self.bootstrap(states, actions)
        for i in range(len(self.policies)):
            print('training net ', str(i))
            self.policies[i].train(data_x[i], data_u[i], epochs, batch_size, split_ratio)
    def forward(self, x):
        N = len(self.policies)
        out = []
        mean = 0
        std  = 0
        for i in range(N):
            out_i = self.policies[i](x)
            out.append(out_i)
            mean += (1/N)*out_i
        
        for i in range(N):
            std += (1/(N-1))*torch.sqrt((out[i] - mean)**2)
        return mean, std
            
            

                    

