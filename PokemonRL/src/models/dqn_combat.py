import torch
import torch.nn as nn
import torch.nn.functional as F


class CombatDQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(CombatDQN, self).__init__()
        # input_dim: Cantidad de stats que ve (HP propio, HP enemigo, tipos...)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.head = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)
