'''
    The linear dynamics model of a spring damper system is tested here.
'''
import sys
sys.path.append('../core')

import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
from scipy.linalg import solve_discrete_are

from Utils import TerminalComponents
from Utils import ReplayBuffer
from StabilizingController import SMPC
from SafetyFilter import MPSafetyFilter
from LearningController import LearningController
from ParametricPolicy import ParametricPolicy

from casadi import *
import matplotlib.pyplot as plt
import polytope as pc
import dill
import torch
import ipdb
def simulate_dynamics(dynamics, x0, u, steps = 10000, plot = True):
    '''
        Assumes a 2D state space
    '''
    x_current = x0 
       
    if (plot):
        
        # state buffer
        x_hist1 = [x0[0,0]]
        x_hist2 = [x0[1,0]]

        # plotting options
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        Ln, = ax.plot(x_hist1, x_hist2)
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        plt.title('System Dynamics')
        plt.xlabel('x1 (position)')
        plt.ylabel('x2 (velocity)')
        plt.ion()
        plt.show()
    

    for i in range(steps):
        x_next = dynamics(x_current, u)
        x_current = x_next

        if (plot):
            x_hist1.append(x_next[0,0])
            x_hist2.append(x_next[1,0])
        
            Ln.set_xdata(x_hist1)
            Ln.set_ydata(x_hist2)
            plt.pause(0.01)

    return x_current   

def ellipse_contoure(P, alpha):
    
    '''
       returns the ellpise contoure defined by the matrix P and the levelset alpha 
    '''
    res = 1000
    L = np.linalg.cholesky(P/alpha)
    t = np.linspace(0, 2*np.pi,res)
    z = np.concatenate([np.cos(t).reshape((1,res)), np.sin(t).reshape((1,res))], axis = 0)
    ellipse = np.linalg.solve(L,z)
    return ellipse

def load_model(filename):
    with open(filename, 'rb') as input:
        return dill.load(input)
    return 0
def main():
    
    ''' 
        Files and directories
    '''
    models_path = '../models/controllers/'
    stabilizing_controller_path = models_path + 'stabilizing_controller_linear.pkl' 
    safety_filter_path = models_path + 'safety_filter_linear.pkl' 
    learning_controller_path = models_path + 'learning_controller.pkl' 
    data_path = '../data/'
    '''
        Import the created controller
    '''
    stabilizing_controller = load_model(stabilizing_controller_path)
    learning_controller = load_model(learning_controller_path)
    safety_filter       = load_model(safety_filter_path)
    

    '''
        Define options and flags
    '''
    visualize_constraints = True
    sim_steps             = 100
    
    #initial condition
    x0                    = np.array([-1,1]).reshape(2,1)
    
 
    '''
       read the dynamics model 
    '''
    dynamics = stabilizing_controller.dynamics
              
    '''
        read the input and state constraints
    '''
    x_dim = stabilizing_controller.x_dim
    u_dim = stabilizing_controller.u_dim
    G_x = stabilizing_controller.G_x
    f_x = stabilizing_controller.f_x
 
    G_u = stabilizing_controller.G_u
    f_u = stabilizing_controller.f_u
    
    alpha_stabilizing = stabilizing_controller.alpha
    P = stabilizing_controller.P
    alpha_filter = safety_filter.alpha

    print("stabilizing controller level set: ", alpha_stabilizing)
    print("filter level set: ", alpha_filter)

    '''
        visualize constraints and the ellipsoidal set
    '''

    # defining the ellipse contours for visualization
    terminal_set_stabilizing = ellipse_contoure(P, alpha_stabilizing)
    terminal_set_filter = ellipse_contoure(P, alpha_filter)
    if (visualize_constraints):

        # defining the state  polytopic constraints
        poly = pc.Polytope(G_x, f_x)

        # plotting
        poly.plot(color = 'pink')
        plt.plot(terminal_set_stabilizing[0,:], terminal_set_stabilizing[1,:])
        plt.plot(terminal_set_filter[0,:], terminal_set_filter[1,:])
        plt.plot(x0[0], x0[1], 'go')
        plt.title('Constriants')
        plt.xlabel('x1 (position)')
        plt.ylabel('x2 (velocity)')
        plt.legend(['terminal set (controller)','terminal set (filter)','initial condition','state constriants'])
        plt.show()


    
    # set up the controllers
    stabilizing_controller.setup()
    safety_filter.setup() 
    
    # instantiate a replay buffer for data storage
    buffer = ReplayBuffer(state_dim = x_dim, act_dim = u_dim)

    # create a grid of the state space
    resolution = 0.1 
    x_max = 2
    x_min = -2
    N = int((x_max-x_min)//resolution)
    x1 = np.linspace(x_min,x_max,N)
    x2 = np.linspace(x_min,x_max,N)
    x1, x2 = np.meshgrid(x1,x2)
    x0_mesh = np.stack([x1,x2], axis = 2)
    
    for i in range(N-1):
        for j in range(N-1):
            
            x0 = x0_mesh[i,j,:].reshape(x_dim,1)

            # Simulate the controller
            x_current = x0
            terminal_set_stabilizing_scaled = terminal_set_stabilizing # the scaled terminal set for the terminal state
    
            # state buffer
            x_hist1 = [x_current[0,0]]
            x_hist2 = [x_current[1,0]]

        
            # plotting terminal set for the stabilizing controller
            elp1 = terminal_set_stabilizing[0,:].tolist() 
            elp2 = terminal_set_stabilizing[1,:].tolist() 
    
            # plotting the scaled terminal set for the stabilizing controller
            elp1_N = terminal_set_stabilizing_scaled[0,:].tolist()
            elp2_N = terminal_set_stabilizing_scaled[1,:].tolist()

            # plotting the terminal set for the safety filter
            elp1_s = terminal_set_filter[0,:].tolist()
            elp2_s = terminal_set_filter[1,:].tolist()



            # plotting planned trajectory
            traj1 = [x0[0]]
            traj2 = [x0[1]]

            # plotting options
            poly.plot(color = 'pink') 
            dyn,  = plt.plot(x_hist1, x_hist2, '-o')
            elp,  = plt.plot(elp1, elp2)
            elp_N, = plt.plot(elp1_N, elp2_N)
            elp_s, = plt.plot(elp1_s, elp2_s)
            tra,  = plt.plot(traj1, traj2, '-o') 
     
    
            plt.title('System Dynamics')
            plt.xlabel('x1 (position)')
            plt.ylabel('x2 (velocity)') 
    
            plt.legend(['true trajectory',
                        'controller terminal set', 
                        'scaled terminal set', 
                        'filter terminal set', 
                        'planned trajectory',
                        'state constraints'])
            #plt.ion()
            #plt.show()

            for k in range(sim_steps):
                try:
                    slack_sol = stabilizing_controller.check_slack(x_current) 
                    u0, traj = stabilizing_controller.solve(x_current, slack_sol)

                except:
                    break
                buffer.store(x_current, u0) 
                terminal_set_stabilizing_scaled = ellipse_contoure(P, stabilizing_controller.terminalset_scaled(traj[-x_dim,:]))
             
                traj1 = traj[:,0].tolist()
                traj2 = traj[:,1].tolist()
            
                elp1_N = terminal_set_stabilizing_scaled[0,:].tolist()
                elp2_N = terminal_set_stabilizing_scaled[1,:].tolist()
        
                x_next =  simulate_dynamics(dynamics,x_current, u = u0, steps = 1, plot = False)

                x_hist1.append(x_next[0,0])
                x_hist2.append(x_next[1,0])
                
                elp1_s = terminal_set_filter[0,:].tolist()
                elp2_s = terminal_set_filter[1,:].tolist()

                tra.set_xdata(traj1)
                tra.set_ydata(traj2)
                dyn.set_xdata(x_hist1[:-1])
                dyn.set_ydata(x_hist2[:-1])
                elp.set_xdata(elp1)
                elp.set_ydata(elp2)
                elp_N.set_xdata(elp1_N)
                elp_N.set_ydata(elp2_N)
                elp_s.set_xdata(elp1_s)
                elp_s.set_ydata(elp2_s)

                #plt.pause(0.01)
                x_current = x_next
            
            #plt.close('all')
            #plt.ioff()
    
    data = buffer.get()
    states = data['state']
    inputs = data['act']
    torch.save(states, data_path + 'states.pt')
    torch.save(inputs, data_path + 'inputs.pt')



if __name__ == "__main__":
    main()
