import sys
sys.path.append('../core')
from ParametricPolicy import ParametricPolicy
from ParametricPolicyEnsemble import ParametricPolicyEnsemble
import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import ipdb
@torch.no_grad()
def main():
    '''
        Paths
    '''
    data_path    = '../data/'
    network_path = '../models/networks/'
    
    '''
        Load the data
    '''
    states = torch.load(data_path+'states_linear.pt')
    inputs = torch.load(data_path+'inputs_linear.pt')
    
    x_dim = states.shape[1]
    u_dim = inputs.shape[1]
    u_max = 1
    u_min = -1
    net_count = 4

    '''
        Load the trained network
    '''
    policy = ParametricPolicyEnsemble(net_count, x_dim, u_dim, u_max, u_min, 3, 30)
    policy.load_state_dict(torch.load(network_path+'policy_linear_ensemble.pkl'))
    
    network_out_mean, network_out_std = policy(states)
    '''
        Visualize the data and the network output
    '''
    
    states = states.numpy()
    inputs = inputs.numpy()
    network_out_mean = network_out_mean.numpy()
    network_out_std = network_out_std.numpy()

    '''
        plot a 3d surface of the data
    '''
    n = 4000
    fig1 = plt.figure()
    ax  = plt.axes(projection = '3d')
    ax.scatter3D(states[:n,0], states[:n,1], inputs[:n])
    ax.set_title('MPC policy')
     
    fig = plt.figure() 
    ax  = plt.axes(projection = '3d')
    ax.scatter3D(states[:n,0], states[:n,1], network_out_mean[:n]+3*network_out_std[:n], 'Blue')
    ax.scatter3D(states[:n,0], states[:n,1], network_out_mean[:n]-3*network_out_std[:n], 'Blue')
    ax.scatter3D(states[:n,0], states[:n,1], network_out_mean[:n], 'Green')
    ax.set_title('Approximate Policy')
    plt.show()

if __name__ == '__main__':
    main()
