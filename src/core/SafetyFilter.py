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
import dill
import ipdb

class MPSafetyFilter:
    def __init__(self,x_dim,u_dim, P, G_x, f_x, G_u, f_u, alpha, dynamics, N = 20):
        
        '''
            x_dim       : state dimension
            u_dim       : input dimension 
            G_x         : state constraint matrix
            f_x         : rhs of state constraints
            G_u         : inputs constraint matrix
            f_u         : rhs of input constraints
            alpha       : ellipsoidal level-set defined by matrix P
            N           : horizon length
        '''
        
        self.P     = P
        self.G_x   = G_x
        self.f_x   = f_x
        self.G_u   = G_u
        self.f_u   = f_u
        self.alpha = alpha
        self.N     = N
        self.opti   = casadi.Opti() 
        
        # state and action dimensions:
        self.x_dim = x_dim
        self.u_dim = u_dim
        
        # solver variable/parameter pointers for performance optimization
        self.u     = None
        self.x     = None
        self.u_L   = None

        # solver tolerance
        self.tol = 1e-8
 
        # define a dynamics callback 
        self.dynamics = dynamics

        # define the Safety Filter status
        self.status = {'feasible': True, 'setup': False}
    
    def setup(self):

        ''' 
             setup the optimization problem proposed in casadi opti.
        '''
        
        self.status['setup'] = True

        N       = self.N
        x_dim   = self.x_dim
        u_dim   = self.u_dim
        
        x0    = self.opti.parameter(x_dim,1)          # initial value
        u_L   = self.opti.parameter(u_dim,1)          # learning control input 
        x     = self.opti.variable(N*x_dim,1)         # N states concatenated into 1 vector
        u     = self.opti.variable((N-1)*u_dim,1)     # N inputs concatenated into 1 vector 
        
        # assign to instance variables for global access
        self.x0      = x0
        self.u       = u
        self.x       = x
        self.u_L     = u_L

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
         
        self.opti.solver("ipopt", p_opts, s_opts)
        
        '''
            cost function 
        '''
        
        self.opti.minimize((u[0:u_dim] - u_L).T@(u[0:u_dim] - u_L))
         

        '''
            dynamics constraint
        '''
        
        self.opti.subject_to(x[0:x_dim]  == x0)
        for i in range(1,N):

            self.opti.subject_to(x[i*x_dim:(i+1)*x_dim] == self.dynamics(x[(i-1)*x_dim:i*x_dim],
                                                                         u[(i-1)*u_dim:i*u_dim]))
           
        '''
            state constraints with the slack variables
        '''
        
        for i in range(0,N-1):
            self.opti.subject_to(self.G_x@x[i*x_dim:(i+1)*x_dim] <= self.f_x) 
         
        '''
            input constraints 
        '''
        for i in range(0,N-1):
            self.opti.subject_to(self.G_u@u[i*u_dim:(i+1)*u_dim] <= self.f_u) 

        '''
            terminal constraint
        '''
        
        self.opti.subject_to(x[-x_dim:].T@self.P@x[-x_dim:] <= self.alpha) 
        pass

    def solve(self, x0, u_L):
        
        # solve
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.u_L, u_L) 
        try:
            sol = self.opti.solve()
            self.status['feasible'] = True
        except:
            print('Safety filter is still infeasible. Switching back to the stabilizing controller')
            self.status['feasible'] = False

        # return the control input along with the planned trajectory
        u0    = sol.value(self.u)[:self.u_dim].reshape((self.u_dim,1))
        
        return u0
    
    def save(self, filename):
        if self.status['setup'] == False:
            with open(filename, 'wb') as output:
                dill.dump(self, output)
        else:
            raise Exception('must save the controller before setting it up. Casadi variables cannot be pickled.')

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['opti']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.opti = casadi.Opti()

