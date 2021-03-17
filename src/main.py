'''
    The linear dynamics model of a spring damper system is tested here.
'''
import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
from scipy.linalg import solve_discrete_are
from LMPC import LinearMPC
import matplotlib.pyplot as plt
import polytope as pc
import ipdb

def simulate_dynamics(A,B,x0, u=0, steps = 10000, plot = True):
    '''
        Assumes a 2D state space
    '''
    assert A.shape[0] == 2

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
    
    if u == 0:
        u = np.zeros((B.shape[1], 1))

    for i in range(steps):
        x_next = A@x_current  
        x_current = x_next

        if (plot):
            x_hist1.append(x_next[0,0])
            x_hist2.append(x_next[1,0])
        
            Ln.set_xdata(x_hist1)
            Ln.set_ydata(x_hist2)
            plt.pause(0.01)

    return x_current


def main():
    
    '''
        Define options and flags
    '''
    visualize_constraints = True
    sim_steps             = 1000
    N                     = 30
    
    #constraints

    x_min_pos             = -1
    x_max_pos             = 1
    x_min_vel             = -1
    x_max_vel             = 1
    
    u_min                 = -0.2
    u_max                 =  0.2
    
     
    #initial condition
    x0                    = np.array([-2,-2]).reshape(2,1)
    
    # system parameters
    k                     = 1       
    c                     = 1       
    m                     = 0.5
    
    
    # stage cost matrices
    Q = np.array([[1, 0], [0, 1]]) #state quadratic weights        
    R = 10                         #input quadratic weights  
    S     = np.eye(f_x.shape[0])   #slack quadratic weights
    gamma = 1                      #slack linear weight

    
    '''
        Define the matrices 
    '''
        
    A_c = np.array([[0,1], [-k/m, -c/m]])     # continous time dynamics matrix
    B_c = np.array([0,1/m], ndmin = 1)        # continous time input matrix
    B_c = B_c.reshape((2,1))
    
    x_dim = A_c.shape[1]
    u_dim = B_c.shape[1]

    '''
        Apply exact discretization to the system.
    '''
    
    freq = 50           # sampling frequecy in Hz
    T    = 1/freq       # sampling time
    
    A = expm(T*A_c)
    B  = np.linalg.solve(A_c, (A - np.eye(A.shape[0]))@B_c)

    assert A.shape[0] == 2 and A.shape[1] == 2
    assert B.shape[0] == 2 and B.shape[1] == 1
    
    '''
        Simulate dynamics to check if it works as expected
    '''

    #x0 = np.array([1,0]).reshape((2,1))
    #simulate_dynamics(A,B, x0, steps = 20)
    

    '''
        Check if the system is inherently stable
    '''

    eig_vals = eig(A)[0]
    stable   = True
    for eig_val in eig_vals:
        if (np.iscomplex(eig_val)):
            if (np.absolute(eig_val) >= 1):
                stable = False
        elif(np.abs(eig_val)>=0.99999):
            stable = False
    print("system stable: ", str(stable))

    '''
        Define the MPC stage/terminal cost weights
    '''
    
                    
    P = solve_discrete_are(A, B, Q, R)    # infinite LQR weight penalty
    '''
        Define the LQR state feedback matrix
    '''
    
    K = np.linalg.solve(R + B.T@P@B, -B.T@P@A)
        
    '''
        Simulate closed loop dynamics to check if  K  works as expected
    '''
    #x0 = np.array([2,0]).reshape((2,1))
    #B_k = B@K
    #simulate_dynamics(A + B_k,B, x0, steps = 100000)



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
    
    alpha_opt = 0
    N_const   = len(f_x_u) # number of constraints
    
    for i in range(N_const):
        F = G_x_u[i,:].reshape((1,x_dim))
        alpha = (f_x_u[i]**2)/(F@np.linalg.inv(P)@F.T)
        alpha = alpha.flatten()[0]
        if i == 0:
            alpha_opt = alpha
        elif(alpha<alpha_opt):
            alpha_opt = alpha
    print("optimal level set: ", alpha_opt)
    assert alpha_opt > 0

    '''
        visualize constraints and the ellipsoidal set
    '''
   
    # defining the ellipse coordinates
    res = 100
    L = np.linalg.cholesky(P/alpha_opt)
    t = np.linspace(0, 2*np.pi,res)
    z = np.concatenate([np.cos(t).reshape((1,res)), np.sin(t).reshape((1,res))], axis = 0)
    ellipse = np.linalg.solve(L,z)
    
    if (visualize_constraints):    
        # defining the state  polytopic constraints
        p = pc.Polytope(np.concatenate([G_x, G_x_u], axis = 0),
                        np.concatenate([f_x, f_x_u], axis = 0))
        # plotting
        p.plot(color = 'pink')
        plt.plot(ellipse[0,:], ellipse[1,:])
        plt.title('Constriants')
        plt.xlabel('x1 (position)')
        plt.ylabel('x2 (velocity)')
        plt.show()


    ''' 
        Define the slack linear and quadratic stage costs
    '''
    
    assert S.shape[0] == f_x.shape[0]
    
    '''
        Instantiate the LinearMPC model
    '''
    controller = LinearMPC(A,B,Q,R,P,S,gamma,G_x,f_x,G_u,f_u,alpha_opt, N)
    controller.setup()
    
    '''
        Simulate the controller
    '''

    x_current = x0
    # state buffer
    x_hist1 = [x_current[0,0]]
    x_hist2 = [x_current[1,0]]

        
    # plotting terminal set
    elp1 = ellipse[0,:].tolist() 
    elp2 = ellipse[1,:].tolist() 
    
    # plotting planned trajectory
    traj1 = []
    traj2 = []

    # plotting options
    fig  = plt.figure()
    ax    = fig.add_subplot(111)
    
    dyn,  = ax.plot(x_hist1, x_hist2, '-o')
    elp,  = ax.plot(elp1, elp2)
    tra,  = ax.plot(traj1, traj2, '-o') 
    
    #ax.set_xlim([-10,10])
    #ax.set_ylim([-10,10])
    
    plt.title('System Dynamics')
    plt.xlabel('x1 (position)')
    plt.ylabel('x2 (velocity)') 
    ax.legend(['true trajectory', 'terminal set', 'planned trajectory'])
    plt.ion()
    plt.show()

    for i in range(sim_steps):
        u, traj = controller.solve(x_current)
        x_next =  simulate_dynamics(A,B,x_current, u = u, steps = 1, plot = False)
        x_current = x_next
        
        x_hist1.append(x_next[0,0])
        x_hist2.append(x_next[1,0])
        
        traj1 = traj[:,0].tolist()
        traj2 = traj[:,1].tolist()

        dyn.set_xdata(x_hist1)
        dyn.set_ydata(x_hist2)
        elp.set_xdata(elp1)
        elp.set_ydata(elp2)
        tra.set_xdata(traj1)
        tra.set_ydata(traj2)
        plt.pause(0.5)


if __name__ == "__main__":
    main()
