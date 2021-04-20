import sys
sys.path.append('../core')
from ParametricPolicy import ParametricPolicy
from ParametricPolicyEnsemble import ParametricPolicyEnsemble
import numpy as np
import torch
import dill
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mayavi import mlab
import ipdb
@torch.no_grad()
def main():
    '''
        Flags
    '''
    view = '2D'


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
    policy = torch.load(network_path+'policy_linear_ensemble.pkl')
    network_out_mean, network_out_std = policy(states)
    '''
        Visualize the data and the network output in 3D
    '''
    
    states = states.numpy()
    inputs = inputs.numpy().flatten()
    network_out_mean = network_out_mean.numpy().flatten()
    network_out_std = network_out_std.numpy().flatten()

    if view == '3D':

        # MPC 
        pts_MPC = mlab.points3d(states[:,0], states[:,1], inputs)
        mesh_MPC = mlab.pipeline.delaunay2d(pts_MPC)
        pts_MPC.remove()
        surf_MPC = mlab.pipeline.surface(mesh_MPC, color = (0,0,1))

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

    if view == '2D':
                     
        # visualize for a constant x1, one with a lot of data and one with little data
        idx = np.where(states[:,1] == -1)[0]
        
        _, idx_unique = np.unique(states[:,0], return_index = True)
        
        def filteration(single_idx):
            return single_idx in idx_unique

        idx  = list(filter(filteration,idx))
        
        states_dense  = states[idx,:]
        states_sparse = states[idx,:]
        
               
        inputs_dense  = inputs[idx]
        network_out_mean_dense = network_out_mean[idx] 
        network_out_std_dense = network_out_std[idx]

        sorted_zipped = sorted(zip(states_dense[:,0], inputs_dense, network_out_mean_dense, network_out_std_dense))
        states_dense  = np.array([x for x,_,_,_ in sorted_zipped])
        inputs_dense  = np.array([u for _,u,_,_ in sorted_zipped])
        network_out_mean_dense  = np.array([mean for _,_,mean,_ in sorted_zipped])
        network_out_std_dense  = np.array([std for _,_,_,std in sorted_zipped])

        UCB = network_out_mean_dense + 3*network_out_std_dense
        LCB = network_out_mean_dense - 3*network_out_std_dense
        
        plt.plot(states_dense, inputs_dense, '-ob')
        plt.plot(states_dense, network_out_mean_dense, '-og')
        plt.fill_between(states_dense, LCB, UCB, color ='r')
        plt.legend(['True MPC Policy', 'Estimated MPC Policy', 'Confidence Bounds'])
        plt.title('Learnt MPC Policy at x2 = const')
        plt.xlabel('x1')
        plt.ylabel('u')
        plt.savefig('slice.png')
        plt.show()





 
if __name__ == '__main__':
    main()
