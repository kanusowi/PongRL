import numpy as np
import random
import torch

def set_global_seeds(seed_value):
    np.random.seed(seed_value)  # NumPy
    random.seed(seed_value)     # Python
    torch.manual_seed(seed_value) # PyTorch
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) # Seed for CUDA
        # For potentially greater (but possibly slower) reproducibility with CUDA:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Global seeds set to: {seed_value}")

if __name__ == '__main__':
    set_global_seeds(42)    
    print(f"NumPy random float (post-seeding): {np.random.rand()}")
    print(f"Python random int (post-seeding): {random.randint(0, 100)}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA random tensor (example, post-seeding): {torch.cuda.FloatTensor(1).normal_()}")
    else:
        print(f"PyTorch CPU random tensor (example, post-seeding): {torch.FloatTensor(1).normal_()}")
