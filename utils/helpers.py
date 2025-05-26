# utils/helpers.py

"""
Placeholder for utility functions that might be useful across the project.
For example:
- Custom plotting functions beyond the basic one in train.py.
- Specific data preprocessing or transformation routines if needed for more complex states.
- Functions for managing seeds or environment configurations globally.
"""

import numpy as np
import torch

def set_global_seeds(seed):
    """Sets random seeds for major libraries to ensure reproducibility."""
    np.random.seed(seed)
    random.seed(seed) # Python's built-in random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        # Potentially add cudnn determinism flags, but they can slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def example_utility_function():
    print("This is an example utility function from utils.helpers.")

if __name__ == '__main__':
    set_global_seeds(42)
    example_utility_function()
    print("Seeds set and example utility ran.")
