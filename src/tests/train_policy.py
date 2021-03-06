import sys
sys.path.append('../core')

from ParametricPolicy import ParametricPolicy

import numpy as np
import torch
import ipdb

def main():

    data_path    = '../data/'
    network_path = '../models/networks/' 
    states = torch.load(data_path+'states_linear.pt')
    inputs = torch.load(data_path+'inputs_linear.pt')
    
    x_dim = states.shape[1]
    u_dim = inputs.shape[1]
    u_max = 1
    u_min = -1

    policy = ParametricPolicy(x_dim, u_dim, u_max, u_min, 3, 30) 
    policy.train(states,inputs)
    torch.save(policy.state_dict(), network_path + 'policy_linear.pkl') 

if __name__ == '__main__':
    main()
