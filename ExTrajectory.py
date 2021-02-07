import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

class ExpertTraj:
    def __init__(self, d_tensor, d_option, state_dim, train_ratio = 0.8):
        '''d_tensor -> traj_tensor; d_option -> option'''
        sha = d_tensor.shape[0]
        #option index matrix; time 
        self.train_all = d_tensor[:int(train_ratio*sha)]
        self.train_state = self.train_all[:, :state_dim]
        self.train_action = self.train_all[:, state_dim:]
        self.n_transitions = self.train_all.shape[0]
        self.train_options = d_option[:int(train_ratio*sha)]

        self.val_all = d_tensor[int(train_ratio*sha):]
        self.val_state = self.val_all[:, :state_dim]
        self.val_action = self.val_all[:, state_dim:]
        self.val_options = d_option[int(train_ratio*sha):]

    def eval(self):
        return np.array(self.val_state.detach()), np.array(self.val_action.detach()), np.array(self.val_options.detach())

    def get_train(self):
        return np.array(self.train_state.detach()), np.array(self.train_action.detach()), np.array(self.train_options.detach())

    def train(self):
        return self.train_state, self.train_action, self.train_options 

    def train_initial(self):
        return self.train_all

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        return self.train_state[indexes], self.train_action[indexes], self.train_options[indexes]
    
    def sample_val(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        return self.val_state[indexes], self.val_action[indexes], self.val_options[indexes]
