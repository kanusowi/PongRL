import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FC1_UNITS, FC2_UNITS

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # nn
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))  
        return self.fc3(x) # linear out