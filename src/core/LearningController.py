import numpy as np

class LearningController:
    def __init__(self, x_dim, u_dim, u_max, u_min):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.u_max = u_max
        self.u_min = u_min
    
    def get_action(self, x):

        '''    
             generates an input drawn from a uniform pdf between u_max and u_min
         
        '''
        u = np.random.uniform(low = self.u_min, high = self.u_max, size = (self.u_dim,1))
        u = u.reshape(self.u_dim,1)
        return u  
