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

import matplotlib.pyplot as plt
import polytope as pc
import dill
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

def exact_discretization(A_c, B_c, T):
    A = expm(T*A_c)
    B  = np.linalg.solve(A_c, (A - np.eye(A.shape[0]))@B_c)
    return A, B

def isstable(A):
    eig_vals = eig(A)[0]
    stable   = True
    for eig_val in eig_vals:
        if (np.iscomplex(eig_val)):
            if (np.absolute(eig_val) >= 1):
                stable = False
        elif(np.abs(eig_val)>=0.99999):
            stable = False
    return stable

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



def dynamics_wrapper(A,B):
    def dynamics(x,u):
        return A@x + B@u
    return dynamics

def main():
    
    ''' 
        Files and directories
    '''
    models_path = '../models/controllers'
    
    '''
        Define options and flags
    '''
    visualize_constraints = True
    sim_steps             = 50000
    N                     = 40
    
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
    x0                    = np.array([-1,-1]).reshape(2,1)
    
    # system parameters
    k                     = 1       
    c                     = 0.1       
    m                     = 0.5
    
    
    x_dim = 2
    u_dim = 1

    # stage cost matrices
    Q = 1*np.array([[1, 0], [0, 1]])  #state quadratic weights        
    R = 10                            #input quadratic weights  
    S     = 10*np.eye(n_x_const)      #slack quadratic weights
    gamma = 100                        #slack linear weight
   
    

    '''
        Define the matrices 
    '''
        
    A_c = np.array([[0,1], [-k/m, -c/m]])     # continous time dynamics matrix
    B_c = np.array([0,1/m], ndmin = 1)        # continous time input matrix
    B_c = B_c.reshape((x_dim,u_dim))
    
    
    '''
        Apply exact discretization to the system.
    '''
    
    freq = 50           # sampling frequecy in Hz
    T    = 1/freq       # sampling time
    
    A, B = exact_discretization(A_c, B_c, T)
    '''
        Define dynamics as a closure nested function to pass to the controller object
    '''
    dynamics = dynamics_wrapper(A,B) 
    
    '''
        Simulate dynamics to check if it works as expected
    '''

    #simulate_dynamics(dynamics, x0, u = np.zeros([u_dim,1]),steps = 20)
    

    '''
        Check if the system is inherently stable
    '''
    stable = isstable(A) 
    print("system stable: ", str(stable))
              
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
     

    stabilizing_controller.save(models_path + 'stabilizing_controller_linear.pkl')
    learning_controller.save(models_path + 'learning_controller.pkl')
    safety_filter.save(models_path + 'safety_filter_linear.pkl')

if __name__ == "__main__":
    main()
