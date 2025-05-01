import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork
from .replay_buffer import ReplayBuffer
from config import (BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, EPS_START, EPS_END, EPS_DECAY, MODEL_SAVE_DIR, DEFAULT_MODEL_NAME)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DQNAgent():
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        
        # random.seed(seed) # self.seed = random.seed(seed) is redundant as random.seed returns None
        random.seed(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) # Seed for all GPUs if using CUDA

        # Hyperparameters (loaded from config.py)
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.lr = LR
        self.update_every = UPDATE_EVERY
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        
        # QNet Init
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Initialize target network with weights copied from the local network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()         
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed, device)        
        self.epsilon = self.eps_start
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                if experiences:
                    return self.learn(experiences, self.gamma) 
        return None

    def act(self, state, eps=None):
        current_epsilon = eps if eps is not None else self.epsilon
        
        if random.random() > current_epsilon:
            state_torch = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device) # Convert state to PyTorch tensor
            
            self.qnetwork_local.eval() 
            with torch.no_grad(): 
                action_values = self.qnetwork_local(state_torch) 
            self.qnetwork_local.train() 
            
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma_val):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma_val * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad() 
        loss.backward()            
        self.optimizer.step()      
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
