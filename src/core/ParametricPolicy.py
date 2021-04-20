import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer
import numpy as np 
import ipdb
class MLP(nn.Module):
    '''
        A standard implementation of the neural network
    '''
    def __init__(self, input_dim, output_dim, hidden_size, layers = 2):
        
        super(MLP,self).__init__()

        self.hidden_size  = hidden_size
        input_layer  = nn.Linear(input_dim, hidden_size)
        output_layer = nn.Linear(hidden_size, output_dim) 
        self.layers = nn.ModuleList()
        self.layers.append(input_layer)
        for i in range(layers-2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(output_layer)

class ParametricPolicy(nn.Module):
    def __init__(self, x_dim, u_dim, u_min, u_max, layers, hidden_size = 30, lr = 5e-4):
        '''
            A neural network to learn a parametric policy that matches the 
            MPC behaviour.
        '''
        super(ParametricPolicy, self).__init__()

        # network dimensions and scaling
        input_dim = x_dim
        output_dim = u_dim
        self.u_max = u_max
        self.u_min = u_min
        self.net = MLP(input_dim, output_dim, hidden_size, 2) 
        
        # Optimizer
        self.optimizer = optimizer.Adam(self.net.parameters(), lr)

        # Weight initializations
        self.init_weights()

    def init_weights(self):
        '''
        Weights initialization method
        '''
        for layer in self.net.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        
        '''
        Forward propagation method
        '''
        for i in range(len(self.net.layers)-1):
            x = nn.ReLU()(self.net.layers[i](x))  
        u = nn.Tanh()(self.net.layers[-1](x))
        u = self.map2range(u)
        return u
   
   
    def map2range(self,u):

        u = torch.mul(((u + 1)/2), (self.u_max - self.u_min)) + self.u_min
        return u
       
    def train(self, states, actions, epochs= 300, batch_size = 256, split_ratio = 0.1):

    
        # Training - Validation split
        shuffle_idx = np.arange(states.shape[0])
        np.random.shuffle(shuffle_idx)

        split_idx =  int(states.shape[0]*split_ratio)
        train_idx = shuffle_idx[split_idx:]
        val_idx   = shuffle_idx[0:split_idx:]

        # Training data
        train_states       = states[train_idx,:]
        train_actions      = actions[train_idx,:]
    
        # Validation data
        val_states           = states[val_idx,:]
        val_actions          = actions[val_idx,:]

        # Training Loop
        for epoch in range(epochs):
            for batch in range(train_states.shape[0]//batch_size):
                
                # Randomly select a batch
                batch_idx = np.random.choice(np.arange(train_states.shape[0]), size = batch_size)

                states_batch            = torch.Tensor(train_states[batch_idx,:])
                actions_batch           = torch.Tensor(train_actions[batch_idx,:])

                self.update(states_batch, actions_batch) 
            # Evaluate NNs performance on the validation set
            val_state            = torch.Tensor(val_states)
            val_action           = torch.Tensor(val_actions)

            with torch.no_grad():
                out  = self.forward(train_states)
                loss = nn.L1Loss()(out, train_actions)
            print(f"Epoch: {epoch+1}/{epochs}, train_loss {loss}")
    
    def update(self, states_batch, actions_batch):

        out  = self.forward(states_batch)
        
        loss = nn.L1Loss()(out, actions_batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()




