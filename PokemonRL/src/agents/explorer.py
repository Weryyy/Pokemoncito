import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from src.models.cnn_map import MapCNN


class ExplorerAgent:
    def __init__(self, obs_shape, n_actions, lr=1e-4):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions

        # Redes Neuronales
        self.policy_net = MapCNN(obs_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Hiperparámetros RL
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(
            next_state).unsqueeze(0).to(self.device)

        # --- CORRECCIÓN ROBUSTA TAMBIÉN AQUÍ ---
        action_t = torch.tensor([action], dtype=torch.long).to(self.device)
        action_t = action_t.view(-1, 1)  # Forzar 2D

        reward_t = torch.FloatTensor([reward]).to(self.device)
        done_t = torch.FloatTensor([1.0 if done else 0.0]).to(self.device)

        q_values = self.policy_net(state_t)
        q_val = q_values.gather(1, action_t).squeeze(1)

        with torch.no_grad():
            next_q_values = self.policy_net(next_state_t)
            max_next_q = next_q_values.max(1)[0]
            target = reward_t + (1 - done_t) * self.gamma * max_next_q

        loss = self.loss_fn(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
