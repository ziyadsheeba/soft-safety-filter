import sys
sys.path.append('../core')
from ParametricPolicy import ParametricPolicy
from ParametricPolicyEnsemble import ParametricPolicyEnsemble
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
    inputs = inputs.numpy().flatten()
    network_out_mean = network_out_mean.numpy().flatten()
    network_out_std = network_out_std.numpy().flatten()

    # MPC 
    pts_MPC = mlab.points3d(states[:,0], states[:,1], inputs)
    mesh_MPC = mlab.pipeline.delaunay2d(pts_MPC)
    pts_MPC.remove()
    surf_MPC = mlab.pipeline.surface(mesh_MPC, color = (1,1,1))

    # Neural Network output
    # mean
    pts_NN = mlab.points3d(states[:,0], states[:,1], network_out_mean)
    mesh_NN = mlab.pipeline.delaunay2d(pts_NN)
    pts_NN.remove()
    surf_NN = mlab.pipeline.surface(mesh_NN, color = (0,1,0))
    
    #UCB
    pts_UCB = mlab.points3d(states[:,0], states[:,1], network_out_mean+network_out_std)
    mesh_UCB = mlab.pipeline.delaunay2d(pts_UCB)
    pts_UCB.remove()
    surf_UCB = mlab.pipeline.surface(mesh_UCB, color = (1,0,0))
    
    #LCB
    pts_LCB = mlab.points3d(states[:,0], states[:,1], network_out_mean-network_out_std)
    mesh_LCB = mlab.pipeline.delaunay2d(pts_LCB)
    pts_LCB.remove()
    surf_LCB = mlab.pipeline.surface(mesh_LCB, color = (1,0,0))


    mlab.xlabel('x1')
    mlab.ylabel('x2')
    mlab.zlabel('u')

    mlab.show()
 
if __name__ == '__main__':
    main()
