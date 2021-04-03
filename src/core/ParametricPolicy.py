import torch
import torch.nn as nn
import torch.dunctional as F
import torch.optim as optimizer
import numpy as np 

class NNPolicy:
    def __init__(self, x_dim, u_dim,layers = 2, hidden_size = 10, lr = 5e-4):
        '''
            A neural network to learn a parametric policy that matches the 
            MPC behaviour.
        '''
        super(ANN, self).__init__()

        # network architecture
        input_dim = x_dim
        ouput_dim = u_dim
        self.layer_1 = nn.Linear(input_dim, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_dim)
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
       u = self.layer_2(x)
       return u
   
   def train(self, states, actions, epochs= 100, batch_size = 256, split_ratio = 0.1):
    
       # Training - Validation split
       shuffle_idx = np.arange(states.shape[0])
       np.random.shuffle(shuffle_idx)

       split_idx =  int(state.shape[0]*split_ratio)
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

                out  = self.forward(state_batch)
                assert out.requires_grad == True
                assert out.shape[0] == batch_size

                # Loss Function
                loss = nn.MSELoss()(out, actions_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Evaluate NNs performance on the validation set
            val_state            = torch.Tensor(val_states)
            val_action           = torch.Tensor(val_actions)

            with torch.no_grad():
                out  = self.forward(val_states)
                loss = nn.MSELoss()(out, val_actions)
            print(f"Epoch: {epoch+1}/{epochs}, val_loss {loss}")




