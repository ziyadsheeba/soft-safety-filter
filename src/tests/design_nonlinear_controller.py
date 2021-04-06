'''
    The nonlinear dynamics model of a spring damper system is tested here.
'''
import sys
sys.path.append('../core')

import numpy as np
from numpy.linalg import eig

import scipy
from scipy.linalg import expm
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov

from Utils import TerminalComponents
from StabilizingController import SMPC
from SafetyFilter import MPSafetyFilter
from LearningController import LearningController

from math import sqrt
import matplotlib.pyplot as plt
import polytope as pc
from casadi import *
import ipdb 

def simulate_dynamics(dynamics, x0, u, steps = 10000, plot = True):
    '''
        Assumes a 2D state space for plotting
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

def dynamics_wrapper(T, m, c, k1, k2, mode = 'sim'):
    
    '''
                implement a discretized model using forward euler.
                Incorporate a non-linear spring force to the system
                a = -(k/m)*x² - (c/m)*v
    '''  
    if mode == 'sim':

        def dynamics(x,u):
            x1_next = x[0] + T*x[1]
            x2_next = x[1] + T*(-(k1/m)*x[0] + (-k2/m)*(x[0]**3) + (-c/m)*x[1] +(1/m)*u[0])
            return np.array([x1_next, x2_next]).reshape(2,1)
    elif mode == 'opt':
        def dynamics(x,u):
        
            x1_next = x[0] + T*x[1]
            x2_next = x[1] + T*(-(k1/m)*x[0] + (-k2/m)*(x[0]**3) + (-c/m)*x[1] +(1/m)*u[0])
            return casadi.vertcat(x1_next, x2_next)
    return dynamics

def ellipse_contoure(P, alpha):

    '''
       returns the ellpise contoure defined by the matrix P and the level-set alpha 
    '''
    res = 1000
    L = np.linalg.cholesky(P/alpha)
    t = np.linspace(0, 2*np.pi,res)
    z = np.concatenate([np.cos(t).reshape((1,res)), np.sin(t).reshape((1,res))], axis = 0)
    ellipse = np.linalg.solve(L,z)
    return ellipse

def main():
    
    ''' 
        Define paths
    '''
    models_path = '../models/controllers/'

    '''
        Define options and flags
    '''
    visualize_constraints = True
    sim_steps             = 1000
    N                     = 60
    
    #constraints

    x_min_pos             = -1
    x_max_pos             = 1
    x_min_vel             = -1
    x_max_vel             = 1
    
    u_min                 = -1
    u_max                 =  1
    
    n_x_const             = 4
    n_u_const             = 2
     
    #initial condition
    x0                    = np.array([-1, 1]).reshape(2,1)
    
    # system parameters
    k1                    = 1         # linear spring factor      
    k2                    = 0.1       # cubic spring factor
    c                     = 0.1       
    m                     = 0.5
    
 
    x_dim = 2   
    u_dim = 1

    # stage cost matrices
    Q = 1*np.array([[1, 0], [0, 1]])           #state quadratic weights        
    R = 20*np.array([1]).reshape(u_dim,u_dim)   #input quadratic weights  
    S = 10*np.eye(n_x_const)                   #slack quadratic weights
    gamma = 1000                               #slack linear weight
   
    
    '''
        Define the dynamics
    '''

    freq = 100          # sampling frequecy in Hz
    T    = 1/freq       # sampling time
    
    
    # dynamics used for optimization must not include any numpy operations
    dynamics = dynamics_wrapper(T, m, c, k1, k2, mode = 'opt') 
    dynamics_sim = dynamics_wrapper(T, m, c, k1, k2, mode = 'sim') 
    
    # simulate the dynamics
    #simulate_dynamics(dynamics_sim, x0 = x0, u = np.zeros([u_dim,1]),steps =  1000)

    '''
        choose an equilibrium point and define the linearized dynamics
        equilibrium point chosen is [0,0]
    '''
    A_c = np.array([[0,1], [-k1/m, -c/m]])     # continous time dynamics matrix
    B_c = np.array([0,1/m], ndmin = 1)        # continous time input matrix
    B_c = B_c.reshape((x_dim,u_dim))
    
    '''
        Determine the discretized linear system dynamics matrices
    '''

    A    = np.array([[1, T],[-k1*T/m, 1-(c*T/m)]]) 
    B    = np.array([0, T/m]).reshape(x_dim, u_dim)  
    
    '''
        Define state and input constraint matrices
        -state constraints: 
            constrain only the position state to lie beteween
            x_min and x_max
        -input constriants:
            constrain the input to lie between u_min and u_max
    '''
   
    G_x = np.array([[1,0], [-1,0] , [0,1], [0,-1]])
    f_x = np.array([x_max_pos,-x_min_pos,x_max_vel, -x_min_vel], ndmin = 1).reshape((4,1))
 
    G_u = np.array([1,-1], ndmin = 1).reshape((2,1))
    f_u = np.array([u_max, -u_min], ndmin = 1).reshape((2,1))
    
    '''
        Compute the terminal set components 
    '''
    dynamics = dynamics_wrapper(T, m, c, k1, k2, mode = 'opt')
    terminal_obj = TerminalComponents(A = A,
                                      B = B, 
                                      Q = Q, 
                                      R = R, 
                                      G_x = G_x, 
                                      f_x = f_x, 
                                      G_u = G_u, 
                                      f_u = f_u, 
                                      dynamics_type = "nonlinear",
                                      dynamics = dynamics)

    alpha_opt_stabilizing, P = terminal_obj.compute_terminal_set(mode = 'input')
    alpha_opt_filter, _ = terminal_obj.compute_terminal_set(mode = 'both') 
    
    print("stabilizing controller level set: ", alpha_opt_stabilizing)
    print("filter level set: ", alpha_opt_filter)
    
    '''
        visualize constraints and the ellipsoidal set
    '''
        
    # defining the ellipse coordinates
    terminal_set_stabilizing = ellipse_contoure(P, alpha_opt_stabilizing) 
    terminal_set_filter = ellipse_contoure(P, alpha_opt_filter) 
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

 
    '''
        Instantiate the soft MPC model, the safety filter and the learning controller
    '''
    stabilizing_controller  = SMPC(Q,R,P,S,gamma,
                                   G_x,f_x,G_u,f_u,
                                   alpha_opt_stabilizing,dynamics, N)

    safety_filter           = MPSafetyFilter(2, 1, P,
                                             G_x, f_x, G_u, f_u,
                                             alpha_opt_filter, dynamics, N)
    learning_controller     = LearningController(x_dim, u_dim, u_max, u_min)
    
    stabilizing_controller.save(models_path + 'stabilizing_controller_nonlinear.pkl')
    learning_controller.save(models_path + 'learning_controller.pkl')
    safety_filter.save(models_path + 'safety_filter_nonlinear.pkl')
 

 

if __name__ == "__main__":
    main()
