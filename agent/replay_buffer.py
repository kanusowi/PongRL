import random
import torch
import numpy as np
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): device to store tensors (cpu or cuda)
        """
        self.memory = deque(maxlen=int(buffer_size)) # Ensure buffer_size is int
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        np.random.seed(seed) # Seed numpy for sampling
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Ensure states are numpy arrays before creating the experience tuple
        e = self.experience(np.array(state), action, reward, np.array(next_state), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) < self.batch_size: # Avoid sampling if not enough experiences
            return None

        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to PyTorch tensors and move to the correct device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device) # Actions are usually indices
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # Dones should be 0 or 1, ensure correct type and shape for broadcasting in Bellman equation
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
