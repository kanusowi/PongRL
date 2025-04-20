import numpy as np
import random
# from collections import namedtuple, deque # Will be used by ReplayBuffer

# PyTorch imports will be added when QNetwork and ReplayBuffer are implemented
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from .model import QNetwork

# Hyperparameters will be imported from config.py
# from config import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, EPS_START, EPS_END, EPS_DECAY

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment using DQN."""

    def __init__(self, state_size, action_size, seed=0,
                 config_params=None): # Pass config object or dict
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): number of actions
            seed (int): random seed for reproducibility
            config_params (dict/object): Contains hyperparameters like LR, GAMMA, etc.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        np.random.seed(seed) # Also seed numpy if used for agent's random choices
        # torch.manual_seed(seed) # When PyTorch components are added

        # Store hyperparameters (example of accessing from a passed config object/dict)
        if config_params:
            self.buffer_size = config_params.get('BUFFER_SIZE', int(1e5))
            self.batch_size = config_params.get('BATCH_SIZE', 64)
            self.gamma = config_params.get('GAMMA', 0.99)
            self.tau = config_params.get('TAU', 1e-3)
            self.lr = config_params.get('LR', 5e-4)
            self.update_every = config_params.get('UPDATE_EVERY', 4)
            self.eps_start = config_params.get('EPS_START', 1.0)
            self.eps_end = config_params.get('EPS_END', 0.01)
            self.eps_decay = config_params.get('EPS_DECAY', 0.995)
        else: # Default values if no config_params provided
            self.buffer_size = int(1e5); self.batch_size = 64; self.gamma = 0.99
            self.tau = 1e-3; self.lr = 5e-4; self.update_every = 4
            self.eps_start = 1.0; self.eps_end = 0.01; self.eps_decay = 0.995

        self.epsilon = self.eps_start # Initial epsilon for exploration

        # Time step counter (for updating network every 'update_every' steps)
        self.t_step = 0

        # Q-Network (Placeholders - to be initialized with actual PyTorch models)
        # self.qnetwork_local = None # QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = None # QNetwork(state_size, action_size, seed).to(device)
        # self.optimizer = None # optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory (Placeholder - to be initialized with ReplayBuffer instance)
        # self.memory = None # ReplayBuffer(self.buffer_size, self.batch_size, seed, device)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # This method will be fully implemented once ReplayBuffer and learn() are ready.
        # 1. Add experience to memory: self.memory.add(state, action, reward, next_state, done)
        # 2. Increment t_step. If t_step % self.update_every == 0:
        # 3.   If len(self.memory) > self.batch_size:
        # 4.     experiences = self.memory.sample()
        # 5.     self.learn(experiences, self.gamma)
        pass # Placeholder for now

    def act(self, state, eps=None):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection. If None, use agent's current epsilon.
        """
        current_epsilon = eps if eps is not None else self.epsilon

        # Epsilon-greedy action selection
        # For now, without a Q-network, it will always explore (random action)
        if random.random() > current_epsilon:
            # Exploit: Choose the best action based on Q-network (Not implemented yet)
            # state_torch = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # self.qnetwork_local.eval() # Set network to evaluation mode
            # with torch.no_grad():
            #     action_values = self.qnetwork_local(state_torch)
            # self.qnetwork_local.train() # Set network back to training mode
            # return np.argmax(action_values.cpu().data.numpy())
            return random.choice(np.arange(self.action_size)) # Fallback if Q-net not ready
        else:
            # Explore: Choose a random action
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update Q-network parameters using a batch of experiences."""
        # This method will be fully implemented with Q-network updates.
        # states, actions, rewards, next_states, dones = experiences
        # 1. Calculate target Q-values (Q_targets) using Bellman equation and target network.
        # 2. Calculate expected Q-values (Q_expected) from local network.
        # 3. Compute loss (e.g., MSE loss between Q_expected and Q_targets).
        # 4. Perform gradient descent step (optimizer.zero_grad(), loss.backward(), optimizer.step()).
        # 5. Soft update target network: self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        pass # Placeholder for now

    def update_epsilon(self):
        """Decay epsilon value for exploration strategy."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # def soft_update(self, local_model, target_model, tau_val):
    #     """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target."""
    #     pass # Placeholder

    # def save_model(self, filepath="qnetwork.pth"):
    #     """Saves the Q-network weights."""
    #     # if self.qnetwork_local: torch.save(self.qnetwork_local.state_dict(), filepath)
    #     pass

    # def load_model(self, filepath="qnetwork.pth"):
    #     """Loads Q-network weights from file."""
    #     # if self.qnetwork_local: self.qnetwork_local.load_state_dict(torch.load(filepath))
    #     # if self.qnetwork_target: self.qnetwork_target.load_state_dict(torch.load(filepath)) # Also update target
    #     pass
