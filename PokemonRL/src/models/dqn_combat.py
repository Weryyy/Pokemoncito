import torch
import torch.nn as nn
import torch.nn.functional as F


class CombatDQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(CombatDQN, self).__init__()
        # input_dim: Cantidad de stats que ve (HP propio, HP enemigo, tipos...)
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.head(x)
