'''
    Implements a linear soft-constrait MPC controller with stability guarantees.
    Uses Casadi with Opti stack
'''
from sys import path
path.append(r"./casadi-py27-v3.5.5")


from casadi import *
import numpy as np
from scipy.linalg import fractional_matrix_power
import math
import ipdb

class SMPC:
    def __init__(self, Q, R, P, S, gamma ,G_x, f_x, G_u, f_u, alpha, dynamics, N = 20):
        
        '''
            x_dim       : state dimension
            u_dim       : input dimension 
            Q           : states stage cost
            R           : input stage cost
            P           : terminal cost
            S           : quadratic stage cost on slack
            gamma       : linear slack penalty weight
            G_x         : state constraint matrix
            f_x         : rhs of state constraints
            G_u         : inputs constraint matrix
            f_u         : rhs of input constraints
            alpha       : terminal set level-set defined by P
            N           : horizon length
            opti_slack  : optimization problem for slack 
            opti_perf   : optimization problem for peformance
        '''

        self.Q     = Q
        self.R     = R
        self.P     = P
        self.S     = S
        self.gamma = gamma
        self.G_x   = G_x
        self.f_x   = f_x
        self.G_u   = G_u
        self.f_u   = f_u
        self.alpha = alpha
        self.N     = N
        self.opti_slack  = casadi.Opti()
        self.opti_perf   = casadi.Opti() 
        
        # state and action dimensions:
        try:
            self.x_dim = Q.shape[0]
        except:
            self.x_dim = 1
        try:
            self.u_dim = R.shape[0]
        except:
            self.u_dim = 1

        # solver variable/parameter pointers for slack optimization 
        self.x0        = None
        self.eps_i     = None        
        self.eps_s     = None
        self.u         = None
        self.x         = None
        self.slack_sol = None # used to warm start 
        # solver variable/parameter pointers for performance optimization
        self.u_p       = None
        self.x_p       = None
        self.eps_i_p   = None
        self.eps_s_p   = None
        
        # solver tolerance
        self.tol = 1e-6
 
        # define a dynamics callback 
        self.dynamics = dynamics
        
        # define the controller status
        self.status = {'enabled': True, 'zero_slack': False}

    def setup(self):

        ''' 
             setup the optimization problem proposed in casadi opti.
             In this function we will set up both optimization problems,
             only the objective will vary between slack and perf.
             uses the l1 norm for exactness
        '''
        N       = self.N
        x_dim   = self.x_dim
        u_dim   = self.u_dim
        eps_dim = self.f_x.shape[0]
        
        # slack
        x0    = self.opti_slack.parameter(x_dim,1)          # initial value as parameter for easy update
        x     = self.opti_slack.variable(N*x_dim,1)         # N states concatenated into 1 vector
        u     = self.opti_slack.variable((N-1)*u_dim,1)     # N inputs concatenated into 1 vector
        eps_i = self.opti_slack.variable((N-1)*eps_dim,1)   # N slacks along the horizon
        eps_s = self.opti_slack.variable(eps_dim,1)         # terminal slack
        
        
        # assign to instance variables for global access
        self.x0      = x0
        self.eps_i   = eps_i        
        self.eps_s   = eps_s
        self.u       = u
        self.x       = x

        # performance
        x0_p     = self.opti_perf.parameter(x_dim,1)          # initial value as parameter for easy update
        x_p      = self.opti_perf.variable(N*x_dim,1)         # N states concatenated into 1 vector
        u_p      = self.opti_perf.variable((N-1)*u_dim,1)     # N inputs concatenated into 1 vector
        eps_i_p  = self.opti_perf.parameter((N-1)*eps_dim,1)  # N slacks along the horizon as a parameter
        eps_s_p  = self.opti_perf.parameter(eps_dim,1)        # terminal slack as a parameter
        
        # assign to instance variables for global acess
        self.x0_p    = x0_p
        self.u_p     = u_p
        self.x_p     = x_p
        self.eps_i_p = eps_i_p
        self.eps_s_p = eps_s_p


        
        '''
            choose solver
        '''
        
        p_opts = {}
        s_opts = {'print_level' : 0,
                  'print_user_options': "no",
                  'print_options_documentation': "no",
                  'print_frequency_iter': 10000,
                  'max_iter' : 100000,
                  'tol'      : self.tol}
         
        self.opti_slack.solver("ipopt", p_opts, s_opts)
        self.opti_perf.solver("ipopt", p_opts, s_opts)
        
        '''
            cost function for the slack variables: opti_slack
        '''
        
        # slack
        S_telda   = np.kron(np.eye(N-1), self.S)
        I_rep     = np.kron(np.ones([N-1,1]), np.eye(eps_s.shape[0]))
        
        ones_i = np.ones([eps_i.shape[1], eps_i.shape[0]])
        ones_s = np.ones([eps_s.shape[1], eps_s.shape[0]])
        
        self.opti_slack.minimize(eps_i.T@S_telda@eps_i + 
                                 self.gamma*ones_i@eps_i +
                                 (I_rep@eps_s).T@S_telda@(I_rep@eps_s) + 
                                 self.gamma*ones_i@(I_rep@eps_s) + 
                                 eps_s.T@self.S@eps_s + 
                                 self.gamma*ones_s@eps_s)
        
        
        # performance
        Q_telda = np.kron(np.eye(N), self.Q)
        R_telda = np.kron(np.eye(N-1), self.R)
        
        self.opti_perf.minimize(x_p.T@Q_telda@x_p + 
                                u_p.T@R_telda@u_p + 
                                x_p[-x_dim:].T@self.P@x_p[-x_dim:])
        '''
            dynamics constraint
        '''
        
        self.opti_slack.subject_to(x[0:x_dim]  == x0)
        self.opti_perf.subject_to(x_p[0:x_dim] == x0_p)
        for i in range(1,N):

            self.opti_slack.subject_to(x[i*x_dim:(i+1)*x_dim] == self.dynamics(x[(i-1)*x_dim:i*x_dim],
                                                                               u[(i-1)*u_dim:i*u_dim]))
           
            self.opti_perf.subject_to(x_p[i*x_dim:(i+1)*x_dim] == self.dynamics(x_p[(i-1)*x_dim:i*x_dim],
                                                                                u_p[(i-1)*u_dim:i*u_dim]))
        '''
            state constraints with the slack variables
        '''
        
        for i in range(0,N-1):
            self.opti_slack.subject_to(self.G_x@x[i*x_dim:(i+1)*x_dim] <= self.f_x +  
                                                                         eps_i[i*eps_dim:(i+1)*eps_dim] + 
                                                                         eps_s) 

            self.opti_perf.subject_to(self.G_x@x_p[i*x_dim:(i+1)*x_dim] <= self.f_x + 
                                                                           eps_i_p[i*eps_dim:(i+1)*eps_dim] + 
                                                                           eps_s_p) 
         
        '''
            input constraints 
        '''
        for i in range(0,N-1):
            self.opti_slack.subject_to(self.G_u@u[i*u_dim:(i+1)*u_dim] <= self.f_u) 
            self.opti_perf.subject_to(self.G_u@u_p[i*u_dim:(i+1)*u_dim] <= self.f_u) 

        '''
            terminal constraint
        '''
        
        self.opti_slack.subject_to(x[-x_dim:].T@self.P@x[-x_dim:] <= self.alpha)
        self.opti_perf.subject_to(x_p[-x_dim:].T@self.P@x_p[-x_dim:] <= self.alpha)
 
        '''
            slack non-negativity constriants
        '''

        self.opti_slack.subject_to(casadi.vec(eps_i)>= 0)
        self.opti_slack.subject_to(casadi.vec(eps_s) >= 0) 
        
        '''
            terminal slack scaling
        '''

        P_inv  = np.linalg.inv(self.P)
        c_squared = np.ones([self.f_x.shape[0], 1])
        

        # formulation as a quadratic constraint  
        for i in range(self.f_x.shape[0]):
            
            c_squared[i] = self.G_x[i,:]@P_inv@self.G_x[i,:].T
            self.opti_slack.subject_to(c_squared[i]*(x[-x_dim:].T@self.P@x[-x_dim:]) <= self.f_x[i]**2 + 
                                                                                    eps_s[i]*eps_s[i] + 
                                                                                    2*self.f_x[i]*eps_s[i])
 
            self.opti_perf.subject_to(c_squared[i]*(x_p[-x_dim:].T@self.P@x_p[-x_dim:]) <= self.f_x[i]**2 + 
                                                                                       eps_s_p[i]*eps_s_p[i] + 
                                                                                       2*self.f_x[i]*eps_s_p[i]) 
        pass

    def check_slack(self,x0):
        
        # solve slack problem
        self.opti_slack.set_value(self.x0, x0)
        slack_sol = self.opti_slack.solve()
         
        # check slack variables
        self.check_hard_feasibility(slack_sol)
        
        return slack_sol

    def solve(self, x0, slack_sol):
    
        # warm start perf problem
        self.opti_perf.set_initial(self.x_p, slack_sol.value(self.x))
        self.opti_perf.set_initial(self.u_p, slack_sol.value(self.u))
        
        # initialize the parameters of the performance optimization
        self.opti_perf.set_value(self.x0_p, x0)
        self.opti_perf.set_value(self.eps_s_p, slack_sol.value(self.eps_s))
        self.opti_perf.set_value(self.eps_i_p, slack_sol.value(self.eps_i))
        
        # solve the performance optimization
        sol = self.opti_perf.solve()

        
        # return the control input along with the planned trajectory
        u_dim = self.u_dim
        x_dim = self.x_dim
        u0    = sol.value(self.u_p)[:u_dim].reshape((u_dim,1))
        traj  = sol.value(self.x_p).reshape((self.N, x_dim))
        
        #print("current state: ", [traj[0,0], traj[0,1]])
        #print("terminal state: ", [traj[-1,0], traj[-1,1]])
        #print("terminal slack: ", sol.value(self.eps_s_p))
        
        return u0, traj
    

    def terminalset_scaled(self, x_N):
        alpha_N = x_N.T@self.P@x_N
        return alpha_N

    def check_hard_feasibility(self, sol):
        '''
            Checks if the slack variables are non-zero.
            This is relavant when integrating this scheme with the safety filter.
            As soon as the optimal slack variables are zero, this controller will 
            be disabled and the (learning controller + safety filter) will be 
            enabled
        '''
        norm = np.linalg.norm(np.concatenate([sol.value(self.eps_i), sol.value(self.eps_s)], axis = 0))
        print("slack norm ", norm)
        if norm < self.tol:
            self.status['zero_slack'] = True
            self.status['enabled'] = False
            print("safety filter enabled. Learning initiated")
        else:
            self.status['zero_slack'] = False
            self.status['enabled'] = True
            print("Stabilizing controller enabled. Learning inhibited")
        pass


