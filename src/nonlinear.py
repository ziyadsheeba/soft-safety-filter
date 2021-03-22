'''
    The linear dynamics model of a spring damper system is tested here.
'''
import numpy as np
from numpy.linalg import eig

from scipy.linalg import expm
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov


from MPC import SMPC
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
        assert x0.shape[0] == 2 and x0.shape[1] == 1 
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
    
    if mode == 'sim':

        def dynamics(x,u):
        
            '''
                implement a discretized model using forward euler.
                Incorporate a non-linear spring force to the system
                a = -(k/m)*x² - (c/m)*v
            '''         
            x1_next = x[0] + T*x[1]
            x2_next = x[1] + T*(-(k1/m)*x[0] + (-k2/m)*(x[0]**3) + (-c/m)*x[1] +(1/m)*u[0][0])

            return np.array([x1_next, x2_next]).reshape(2,1)
    elif mode == 'opt':
        def dynamics(x,u):
        
            '''
                implement a discretized model using forward euler.
                Incorporate a non-linear spring force to the system
                a = -(k/m)*x² - (c/m)*v
            '''         
            x1_next = x[0] + T*x[1]
            x2_next = x[1] + T*(-(k1/m)*x[0] + (-k2/m)*(x[0]**3) + (-c/m)*x[1] +(1/m)*u[0][0])

            return casadi.vertcat(x1_next, x2_next)
    return dynamics

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

def phi_wrapper(dynamics, A, B, K):
    
    def phi(x):
        A_k = A + B@K
        return A_k@x - dynamics(x,K@x) 
    return phi

def main():
    
    '''
        Define options and flags
    '''
    visualize_constraints = True
    sim_steps             = 1000
    N                     = 20
    
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
    x0                    = np.array([-1,1]).reshape(2,1)
    
    # system parameters
    k1                    = 1     # linear spring factor      
    k2                    = 0   # cubic spring factor
    c                     = 0.1       
    m                     = 0.5
    
    
    x_dim = 2   
    u_dim = 1

    # stage cost matrices
    Q = 1*np.array([[1, 0], [0, 1]])        #state quadratic weights        
    R = np.array([10]).reshape(u_dim,u_dim) #input quadratic weights  
    S     = 10*np.eye(n_x_const)               #slack quadratic weights
    gamma = 100                             #slack linear weight
   
    
    '''
        Define the dynamics
    '''

    freq = 200           # sampling frequecy in Hz
    T    = 1/freq       # sampling time
    
    
    # dynamics used for optimization must not include any numpy operations
    dynamics = dynamics_wrapper(T, m, c, k1, k2, mode = 'opt') 
    dynamics_sim = dynamics_wrapper(T, m, c, k1, k2) 
    
    # simulate the dynamics
    #simulate_dynamics(dynamics_sim, x0 = x0, u = np.zeros([u_dim,1]),steps =  1000)

    '''
        choose an equilibrium point and define the linearized dynamics
        equilibrium point chosen is [0,0]
    '''
    
    A = np.array([[1 , T], [-k1*T/m, (1-c*T/m)]])
    B = np.array([0, T/m]).reshape(x_dim, u_dim)

    '''
        Solve DARE for the linearized dynamics 
    ''' 
                    
    P_lqr = solve_discrete_are(A, B, Q, R)    # infinite LQR weight penalty
    
    '''
        Define the LQR state feedback matrix for the linearized dynamics
    '''
    
    K = np.linalg.solve(R + B.T@P_lqr@B, -B.T@P_lqr@A)
    
    '''
        find the eigen values of A + B@K and define c and P 
    '''
    A_k = A + B@K
    eig_vals = eig(A_k)[0]
    eig_max  = max(abs(eig_vals))
    c        = 1 - eig_max**2 - 0.01 
    P       = solve_discrete_lyapunov((1/sqrt(1-c))*A_k, Q + (K.T)@R@K) 
    
    ipdb.set_trace()
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

        - Need to ensure that the terminal controller satisfies the input constraints 
          within the set

        - Solved in closed form using support functions. Check MPC lecture notes 
          chapter 5 slide 62
                        
                        x.T@P@x <= alpha
        
    '''
    G_x_u = G_u@K
    f_x_u = f_u
    
    alpha_1 = 0
    N_const   = len(f_x_u) # number of constraints
    
    for i in range(N_const):
        F = G_x_u[i,:].reshape((1,x_dim))
        alpha = (f_x_u[i]**2)/(F@np.linalg.inv(P)@F.T)
        alpha = alpha.flatten()[0]
        if i == 0:
            alpha_1 = alpha
        elif(alpha<alpha_1):
            alpha_1 = alpha
    print("optimal level set: ", alpha_1)
    assert alpha_1 > 0
    
   

    '''
        solve for the optimal alpha, c1 and c2 according to the algorithm in the coverage control paper
    ''' 
    phi = phi_wrapper(dynamics, A, B, K)
    opti = casadi.Opti()

    p_opts = {}
    s_opts = {'max_iter' : 10000}
    opti.solver('ipopt', p_opts, s_opts)
    
    x     = opti.variable(x_dim,1)
    alpha = opti.variable()
    c1    = opti.variable()
    c2    = opti.variable()
    
    opti.minimize(-alpha)
    opti.subject_to(c1 - 2*c2 == c)
    opti.subject_to((phi(x).T)@P@phi(x) <= c1*x.T@P@x)
    opti.subject_to(x.T@A.T@P@phi(x) <= c2*x.T@P@x)
    opti.subject_to(x.T@P@x <= alpha)
    opti.subject_to(alpha <= alpha_1)
    
    try:
        sol = opti.solve()
        alpha_opt = sol.value(alpha)
        print("solved for optimal alpha: ", alpha_opt)
    except:
        alpha_opt = 0
        print("optimal alpha not found. Terminal set set to zero")
        ipdb.set_trace() 

    '''
        visualize constraints and the ellipsoidal set
    '''
        
    # defining the ellipse coordinates
    terminal_set = ellipse_contoure(P, alpha_opt)
    
    if (visualize_constraints):    
        # defining the state  polytopic constraints
        poly = pc.Polytope(G_x, f_x)
        # plotting
        poly.plot(color = 'pink')
        plt.plot(terminal_set[0,:], terminal_set[1,:])
        plt.plot(x0[0], x0[1], 'go')
        plt.title('Constriants')
        plt.xlabel('x1 (position)')
        plt.ylabel('x2 (velocity)')
        plt.legend(['terminal set','initial condition','state constriants'])
        plt.show()

 
    '''
        Instantiate the soft MPC model
    '''
    controller = SMPC(Q,R,P,S,gamma,
                      G_x,f_x,G_u,f_u,
                      alpha_opt,dynamics, N)
    controller.setup()
    
    '''
        Simulate the controller
    '''

    x_current = x0
    terminal_set_scaled = terminal_set # the scaled terminal set for the terminal state

    # state buffer
    x_hist1 = [x_current[0,0]]
    x_hist2 = [x_current[1,0]]


    # plotting terminal set
    elp1 = terminal_set[0,:].tolist()
    elp2 = terminal_set[1,:].tolist()

    # plotting the scaled terminal set
    elp1_N = terminal_set_scaled[0,:].tolist()
    elp2_N = terminal_set_scaled[1,:].tolist()

    # plotting planned trajectory
    traj1 = [x0[0]]
    traj2 = [x0[1]]

    # plotting options
    poly.plot(color = 'pink')
    dyn,  = plt.plot(x_hist1, x_hist2, '-o')
    elp,  = plt.plot(elp1, elp2)
    elp_N, = plt.plot(elp1_N, elp2_N)
    tra,  = plt.plot(traj1, traj2, '-o')


    plt.title('System Dynamics')
    plt.xlabel('x1 (position)')
    plt.ylabel('x2 (velocity)')

    plt.legend(['true trajectory',
                'terminal set',
                'scaled terminal set',
                'planned trajectory',
                'state constraints'])
    plt.ion()
    plt.show()
    for i in range(sim_steps):
        u0, traj = controller.solve(x_current)
        terminal_set_scaled = ellipse_contoure(P, controller.terminalset_scaled(traj[-x_dim,:]))
        x_next =  simulate_dynamics(dynamics,x_current, u = u0, steps = 1, plot = False)

        x_hist1.append(x_next[0,0])
        x_hist2.append(x_next[1,0])

        traj1 = traj[:,0].tolist()
        traj2 = traj[:,1].tolist()

        elp1_N = terminal_set_scaled[0,:].tolist()
        elp2_N = terminal_set_scaled[1,:].tolist()

        tra.set_xdata(traj1)
        tra.set_ydata(traj2)
        dyn.set_xdata(x_hist1[:-1])
        dyn.set_ydata(x_hist2[:-1])
        elp.set_xdata(elp1)
        elp.set_ydata(elp2)
        elp_N.set_xdata(elp1_N)
        elp_N.set_ydata(elp2_N)
        plt.pause(0.01)

        x_current = x_next
  



if __name__ == "__main__":
    main()
