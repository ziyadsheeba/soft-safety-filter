import numpy as np
from casadi import *
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov

import ipdb
class TerminalComponents:
    def __init__(self,A, B, Q, R, G_x, f_x, G_u, f_u, dynamics_type, dynamics):
        
        '''
            1) dynamics: general nonlinear dynamics in  discrete time
            2) dynamics_type : linear or nonlinear
            3) A       : state transition matrix for linearized dynamics
            4) B       : input matrix for linearized dynamics
            5) Q       : state weight matrix
            6) R       : input weight matrix
            7) G_x     : lhs of state constraint matrix
            8) f_x     : rhs of state constraint matrix
            9) G_u     : lhs of input constraint matrix
            10) f_u     : rhs of input constraint matrix

        '''
        if not (dynamics_type == 'linear') and not (dynamics_type== 'nonlinear'):
           raise Exception("dynamics_type must be either 'linear' or 'nonlinear' ")
        
        self.x_dim    = A.shape[0]
        self.u_dim    = B.shape[1]

        self.A             = A
        self.B             = B
        self.Q             = Q
        self.R             = np.array(R).reshape(self.u_dim,1)
        self.G_x           = G_x
        self.G_u           = G_u
        self.f_x           = f_x
        self.f_u           = f_u
        
        self.dynamics_type = dynamics_type
        self.dynamics = dynamics  
        
                
        self.P = None
        self.K = None
        
        self.eps_Q = 0
        self.eps_R = 0
        
        if dynamics_type == 'linear':
            self.compute_terminal_set = self._compute_invariant_set_linear
        if dynamics_type == 'nonlinear':
            self.compute_terminal_set = self._compute_invariant_set_nonlinear

    def compute_lqr(self):

        if (self.dynamics_type == 'nonlinear'):
            self.eps_Q = 2
            self.eps_R = 2
        else:
            self.eps_Q = 0
            self.eps_R = 0

        eps_I_Q = np.eye(self.x_dim)*self.eps_Q
        eps_I_R = np.eye(self.u_dim)*self.eps_R
        
        P_lqr = solve_discrete_are(self.A, self.B, self.Q + eps_I_Q, self.R + eps_I_R) 
        K_lqr = np.linalg.solve(self.R + self.B.T@P_lqr@self.B, -self.B.T@P_lqr@self.A)  
        
        self.P = P_lqr
        self.K = K_lqr
        pass  
     
    def _compute_invariant_set_linear(self, mode):
        
        if not (mode == 'input') and not (mode == 'both'):
            raise Exception("mode must be either 'input' or 'both' ")

        #if self.dynamics_type == 'nonlinear':
        #    self.compute_lqr_nonlinear(c =0.05)
        #else:
        
        self.compute_lqr()
        if mode == 'input': 
            ''' 
                considers only input constraints
            '''
            G_x_u = self.G_u@self.K
            f_x_u = self.f_u

        if mode == 'both':

            '''
                considers both state and input constraints
            '''
            G_x_u = np.concatenate([self.G_u@self.K, self.G_x], axis = 0)
            f_x_u = np.concatenate([self.f_u, self.f_x], axis = 0)

        alpha_opt = 0
        N_const   = len(f_x_u) 
        for i in range(N_const):
            F = G_x_u[i,:].reshape((1,self.x_dim))
            alpha = (f_x_u[i]**2)/(F@np.linalg.inv(self.P)@F.T)
            alpha = alpha.flatten()[0]
            if i == 0:  
                alpha_opt = alpha
            elif(alpha<alpha_opt):
                alpha_opt = alpha 
        
        return alpha_opt, self.P
    
    def _compute_invariant_set_nonlinear(self, mode):
        alpha_opt, _ = self._compute_invariant_set_linear(mode)
 
        bisection_error = 1e-5
        low = 0
        high = alpha_opt
        while bisection_error <= high-low:
             
            invariant     = self._is_invariant(high)
            suff_decrease = self._sufficient_decrease(high)
            if(not invariant or not suff_decrease):
                high = (high-low)/2 + low 
            else:
                if high == alpha_opt:
                    break
                low_aux = low
                low  = high
                high = high + (high-low_aux)/2 
        alpha_opt = high
        return alpha_opt, self.P  
    def _is_invariant(self, alpha_opt):
        opti = casadi.Opti()
        opti.solver('ipopt')

        x     = opti.variable(self.x_dim,1)
        alpha = opti.parameter()
        f = (self.dynamics(x, self.K@x).T)@self.P@(self.dynamics(x,self.K@x))
        opti.minimize(-f) 
        opti.subject_to(x.T@self.P@x <= alpha)
        opti.set_value(alpha, alpha_opt)
 
        vals = []
        restarts = 10
        
        for i in range(restarts):
            initial = np.random.normal(loc = 0, scale = 10, size = (self.x_dim,1)).reshape(self.x_dim,1)  
            opti.set_initial(x, initial)
            sol = opti.solve()
            vals.append(sol.value(f))
        
        val = max(vals)  
        if val>= alpha_opt:
            invariant = False
        else:
            invariant =  True
        return invariant
    def _sufficient_decrease(self,alpha_opt):
        opti = casadi.Opti()
        opti.solver('ipopt')
        x = opti.variable(self.x_dim,1)
        alpha = opti.parameter()

        if (self.dynamics_type == 'nonlinear'):
            self.eps_Q = 2
            self.eps_R = 2
        else:
            self.eps_Q = 0
            self.eps_R = 0

        eps_I_Q = np.eye(self.x_dim)*self.eps_Q
        eps_I_R = np.eye(self.u_dim)*self.eps_R

        f = self.dynamics(x,self.K@x).T@self.P@self.dynamics(x, self.K@x)  - x.T@(self.P - (self.Q - eps_I_Q) - self.K.T@(self.R-eps_I_R)@self.K)@x
        opti.minimize(-f)
        opti.subject_to(x.T@self.P@x<= alpha_opt)
        opti.set_value(alpha, alpha_opt)

        vals = []
        restarts = 10
        
        for i in range(restarts):
            initial = np.random.normal(loc = 0, scale = 10, size = (self.x_dim,1)).reshape(self.x_dim,1)  
            opti.set_initial(x, initial)
            sol = opti.solve()
            vals.append(sol.value(f))
        
        val = max(vals) 
        if (val <= 1e-7):
            suff_decrease = True
        else:
            suff_decrease = False
        return suff_decrease 
