'''
    The linear dynamics model of a spring damper system is tested here.
'''
import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
from scipy.linalg import solve_discrete_are
from StabilizingController import SMPC
from SafetyFilter import MPSafetyFilter
from LearningController import LearningController
import matplotlib.pyplot as plt
import polytope as pc
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
        Define options and flags
    '''
    visualize_constraints = True
    sim_steps             = 1000
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
    x0                    = np.array([1.3,-1]).reshape(2,1)
    
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
        Solve DARE 
    '''
    
                    
    P = solve_discrete_are(A, B, Q, R)    # infinite LQR weight penalty
    
    '''
        Define the LQR state feedback matrix
    '''
    
    K = np.linalg.solve(R + B.T@P@B, -B.T@P@A)
        
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
        Solve for an ellipsoidal terminal set that respects only input constraints.
        
        - The ellipsoidal terminal set is a lyaponov level set of the system after
          applying the LQR feedback controller

        - Need to ensure that theterminal controller satisfies the input constraints 
          within the set

        - Solved in closed form using support functions. Check MPC lecture notes 
          chapter 5 slide 62
                        
                        x.T@P@x <= alpha
        
    '''
    G_x_u = G_u@K
    f_x_u = f_u
    
    alpha_opt_stabilizing = 0
    N_const   = len(f_x_u) # number of constraints
    
    for i in range(N_const):
        F = G_x_u[i,:].reshape((1,x_dim))
        alpha = (f_x_u[i]**2)/(F@np.linalg.inv(P)@F.T)
        alpha = alpha.flatten()[0]
        if i == 0:
            alpha_opt_stabilizing = alpha
        elif(alpha<alpha_opt_stabilizing):
            alpha_opt_stabilizing = alpha
    print("optimal level set for stabilizing controller: ", alpha_opt_stabilizing)
    assert alpha_opt_stabilizing > 0

    '''
        Solve for an ellipsoidal terminal set that respects only input constraints and state
        constraints for the safety filter.
        
        - The ellipsoidal terminal set is a lyaponov level set of the system after
          applying the LQR feedback controller

        - Need to ensure that theterminal controller satisfies the input constraints 
          within the set

        - Solved in closed form using support functions. Check MPC lecture notes 
          chapter 5 slide 62
                        
                        x.T@P@x <= alpha
        
    '''
    G_x_u = np.concatenate([G_u@K, G_x], axis = 0)
    f_x_u = np.concatenate([f_u, f_x], axis = 0)
    
    alpha_opt_filter = 0
    N_const   = len(f_x_u) # number of constraints
    
    for i in range(N_const):
        F = G_x_u[i,:].reshape((1,x_dim))
        alpha = (f_x_u[i]**2)/(F@np.linalg.inv(P)@F.T)
        alpha = alpha.flatten()[0]
        if i == 0:
            alpha_opt_filter = alpha
        elif(alpha<alpha_opt_filter):
            alpha_opt_filter = alpha

    print("optimal level set for safety filter: ", alpha_opt_filter)
    assert alpha_opt_filter > 0
    
    '''
        visualize constraints and the ellipsoidal set
    ''' 
    # defining the terminal set contoure
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
        plt.legend(['terminal set (stabilizing MPC)', 'terminal set (saftey filter)', 'initial condition','state constriants'])
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
    
    stabilizing_controller.setup()
    safety_filter.setup() 
    '''
        Simulate the controller
    '''

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
    plt.ion()
    plt.show()

    for i in range(sim_steps):

        slack_sol = stabilizing_controller.check_slack(x_current) 
        if stabilizing_controller.status['zero_slack']:
            u_L      = learning_controller.get_action(x_current)
            u_filter = safety_filter.solve(x_current, u_L)
            u0 = u_filter
            traj1 = []
            traj2 = []
            elp1_N = []
            elp2_N = []

        else:
            u0, traj = stabilizing_controller.solve(x_current, slack_sol)
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

        plt.pause(0.01)

        x_current = x_next
        
        if (i+1)%100 == 0:
            print('singular disturbance applied')
            x_current = np.random.uniform(low = -2, high = 2, size = (x_dim,1)).reshape(x_dim,1)
            x_hist1 = [x_current[0,0]]
            x_hist2 = [x_current[1,0]]

if __name__ == "__main__":
    main()
