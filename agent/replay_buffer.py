import random
import torch
import numpy as np
from collections import deque, namedtuple

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        random.seed(seed)
        # np.random.seed(seed) # if using NumPy for any random operations within the buffer
        
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(np.array(state, dtype=np.float32), 
                            action, 
                            reward, 
                            np.array(next_state, dtype=np.float32), 
                            done)
        self.memory.append(e)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None        
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device) # Actions are usually indices, so long type
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
