import torch
import torch.nn as nn
import torch.dunctional as F
import torch.optim as optimizer
import numpy as np 

class ANN:
    def __init__(self, x_dim, u_dim,layers = 2, hidden_size = 10, lr = 5e-4):
        '''
            A neural network to learn a safety materic 
        '''
        super(ANN, self).__init__()

        # network architecture
        input_dim = x_dim + u_dim
        self.layer_1 = nn.Linear(input_dim, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, 1)
        self.layers  = [self.layer_1, self.layer_2]
        
        # Optimizer
        self.optimizer = optimizer.Adam(self.parameters(), lr)

        # Weight initializations
        self.init_weights()


    def init_weights(self):
        '''
        Weights initialization method
        '''
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
       '''
       Forward propagation method
       '''
       x = nn.ReLU()(self.layer_1(x))
       x = self.layer_2(x)
       return x



