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
        
        # reproducibility seed 
        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) # gpu poor :'(

        # hyperparams
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
        
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval() # only for inference, not training
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed, device) # replay memory
        self.epsilon = self.eps_start
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) >= self.batch_size: # check enough samples before learning
                experiences = self.memory.sample()
                if experiences:
                    self.learn(experiences, self.gamma)

    def act(self, state, eps=None):
        current_epsilon = eps if eps is not None else self.epsilon        
        if random.random() > current_epsilon:
            state_torch = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)        
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
        
        self.optimizer.zero_grad() # zero the gradients
        loss.backward()     # compute gradients
        self.optimizer.step()  # update local model parameters

        # DDPG # Soft Update 
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def update_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
            
    def soft_update(self, local_model, target_model, tau_val):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau_val*local_param.data + (1.0-tau_val)*target_param.data)

    def save_model(self, filename=DEFAULT_MODEL_NAME):
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
        filepath = os.path.join(MODEL_SAVE_DIR, filename)
        
        if self.qnetwork_local:
            torch.save(self.qnetwork_local.state_dict(), filepath)
            print(f"Agent model saved to {filepath}")
        else:
            print("Error: Q-network local is not initialized. Cannot save model.")

    def load_model(self, filename=DEFAULT_MODEL_NAME):
        filepath = os.path.join(MODEL_SAVE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Error: Model file not found at {filepath}. Cannot load model.")
            return
        map_loc = device
        
        try:
            if self.qnetwork_local:
                self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=map_loc))
                self.qnetwork_local.train()
            if self.qnetwork_target:
                self.qnetwork_target.load_state_dict(torch.load(filepath, map_location=map_loc))
                self.qnetwork_target.eval()
            print(f"Agent model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model weights from {filepath}: {e}")