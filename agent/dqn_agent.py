import numpy as np
import random
# from collections import namedtuple, deque # For ReplayBuffer

import torch
import torch.nn.functional as F # Will be used in learn()
import torch.optim as optim
from .model import QNetwork # Import the actual model
# Import hyperparameters from config.py (or pass as dict)
from config import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, EPS_START, EPS_END, EPS_DECAY

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Use hyperparameters from config.py
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.lr = LR
        self.update_every = UPDATE_EVERY
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Initialize target network with local network's weights
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval() # Target network is only for inference

        # Replay memory (Placeholder - to be initialized with ReplayBuffer instance)
        # self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed, device)

        self.t_step = 0 # For UPDATE_EVERY logic
        self.epsilon = self.eps_start # Initialize epsilon

    def step(self, state, action, reward, next_state, done):
        # (To be implemented fully with ReplayBuffer)
        # self.memory.add(state, action, reward, next_state, done)
        # self.t_step = (self.t_step + 1) % self.update_every
        # if self.t_step == 0:
        #     if len(self.memory) > self.batch_size:
        #         experiences = self.memory.sample()
        #         self.learn(experiences, self.gamma)
        pass

    def act(self, state, eps=None):
        current_epsilon = eps if eps is not None else self.epsilon

        if random.random() > current_epsilon: # Exploit
            # Convert state to PyTorch tensor
            state_torch = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval() # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculation for inference
                action_values = self.qnetwork_local(state_torch)
            self.qnetwork_local.train() # Set model back to training mode
            return np.argmax(action_values.cpu().data.numpy()) # Get action with max Q-value
        else: # Explore
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma_val): # Renamed gamma to gamma_val to avoid conflict
        # (To be implemented fully with ReplayBuffer samples)
        # states, actions, rewards, next_states, dones = experiences
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Q_targets = rewards + (gamma_val * Q_targets_next * (1 - dones))
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        # loss = F.mse_loss(Q_expected, Q_targets)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        pass

    def update_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def soft_update(self, local_model, target_model, tau_val): # Renamed tau parameter
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau_val*local_param.data + (1.0-tau_val)*target_param.data)

    def save_model(self, filepath="qnetwork.pth"): # Added save_model
        if self.qnetwork_local:
            torch.save(self.qnetwork_local.state_dict(), filepath)
            print(f"Model saved to {filepath}")

    def load_model(self, filepath="qnetwork.pth"): # Added load_model
        map_loc = device # ensure loading to correct device
        if self.qnetwork_local:
            self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=map_loc))
        if self.qnetwork_target: # Also load into target network and ensure it's in eval mode
            self.qnetwork_target.load_state_dict(torch.load(filepath, map_location=map_loc))
            self.qnetwork_target.eval()
        print(f"Model loaded from {filepath}")
