import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork
from .replay_buffer import ReplayBuffer # Import the actual ReplayBuffer
# Import hyperparameters from config.py
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

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval() # Target network in eval mode

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed, device) # Initialize ReplayBuffer

        self.t_step = 0 # For UPDATE_EVERY logic
        self.epsilon = self.eps_start # Initialize epsilon

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done) # Save experience

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                if experiences: # Check if sampling was successful
                    self.learn(experiences, self.gamma)

    def act(self, state, eps=None): # act method remains largely the same
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

    def learn(self, experiences, gamma_val): # Renamed gamma to gamma_val
        """Update Q-network parameters using a batch of experiences."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q-values (for next states) from target model
        # .detach() prevents gradients from flowing into the target network
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states using Bellman equation: R + γ * max_a' Q(s', a'; θ_target)
        # For terminal states (done=1), the Q value is just the reward
        Q_targets = rewards + (gamma_val * Q_targets_next * (1 - dones))

        # Get expected Q-values from local model for the actions taken
        # .gather(1, actions) selects the Q-value corresponding to the action taken for each state
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward()           # Compute new gradients
        self.optimizer.step()       # Update network parameters

        # Update target network (soft update)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def update_epsilon(self):
        """Decay epsilon value for exploration strategy."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def soft_update(self, local_model, target_model, tau_val): # Renamed tau parameter
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau_val*local_param.data + (1.0-tau_val)*target_param.data)

    def save_model(self, filepath="qnetwork.pth"): # Renamed from save
        if self.qnetwork_local:
            torch.save(self.qnetwork_local.state_dict(), filepath)
            print(f"Model saved to {filepath}")

    def load_model(self, filepath="qnetwork.pth"): # Renamed from load
        map_loc = device # ensure loading to correct device
        if self.qnetwork_local:
            self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=map_loc))
        if self.qnetwork_target: # Also load into target network and ensure it's in eval mode
            self.qnetwork_target.load_state_dict(torch.load(filepath, map_location=map_loc))
            self.qnetwork_target.eval()
        print(f"Model loaded from {filepath}")
