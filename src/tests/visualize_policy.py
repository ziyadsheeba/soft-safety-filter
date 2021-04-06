import sys
sys.path.append('../core')
from ParametricPolicy import ParametricPolicy
import numpy as np
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mayavi import mlab
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
    u_min = -1
    u_max = 1
    layers = 3

    '''
        Load the trained network
    '''
    policy = ParametricPolicy(x_dim, u_dim, u_max, u_min,layers)
    policy.load_state_dict(torch.load(network_path+'policy_linear.pkl'))
    
    network_out = policy(states)

    '''
        Visualize the data and the network output
    '''
    
    states = states.numpy()
    inputs = inputs.numpy().flatten()
    network_out = network_out.numpy().flatten()


    '''
        plot a 3d surface of the data
    '''
    # MPC 
    pts_MPC = mlab.points3d(states[:,0], states[:,1], inputs)
    mesh_MPC = mlab.pipeline.delaunay2d(pts_MPC)
    pts_MPC.remove()
    surf_MPC = mlab.pipeline.surface(mesh_MPC, color = (1,1,1))
        
    # Neural Network output
    pts_NN = mlab.points3d(states[:,0], states[:,1], network_out)
    mesh_NN = mlab.pipeline.delaunay2d(pts_NN)
    pts_NN.remove()
    surf_NN = mlab.pipeline.surface(mesh_NN, color = (0,1,0))

    mlab.xlabel('x1')
    mlab.ylabel('x2')
    mlab.zlabel('u') 

    mlab.show()
    
if __name__ == '__main__':
    main()
