import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from src.models.dqn_combat import CombatDQN


class TacticianAgent:
    def __init__(self, input_dim, n_actions, lr=1e-3):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions

        self.policy_net = CombatDQN(input_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.95

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        try:
            # 1. Preparar Tensores
            state_t = torch.FloatTensor(state).to(self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)

            next_state_t = torch.FloatTensor(next_state).to(self.device)
            if next_state_t.dim() == 1:
                next_state_t = next_state_t.unsqueeze(0)

            # 2. Preparar Acción (El culpable habitual)
            action_t = torch.tensor(
                [[action]], dtype=torch.long).to(self.device)

            reward_t = torch.FloatTensor([reward]).to(self.device)
            done_t = torch.FloatTensor([1.0 if done else 0.0]).to(self.device)

            # 3. Calcular Q-Values
            q_values = self.policy_net(state_t)

            # --- ZONA DE DEBUG ---
            # Si las dimensiones no coinciden, imprimimos chivatazo antes del error
            if q_values.dim() != action_t.dim():
                print("\n" + "="*30)
                print("🚨 --- DEBUG ERROR DETECTADO --- 🚨")
                print(f"Input Action (raw): {action}")
                print(
                    f"Tensor Q_Values Shape: {q_values.shape} | Dims: {q_values.dim()}")
                print(
                    f"Tensor Action Shape:   {action_t.shape} | Dims: {action_t.dim()}")
                print("La función .gather() va a fallar ahora mismo.")
                print("="*30 + "\n")

            # 4. Gather y Loss
            q_val = q_values.gather(1, action_t).squeeze(1)

            with torch.no_grad():
                next_q_values = self.policy_net(next_state_t)
                max_next_q = next_q_values.max(1)[0]
                target = reward_t + (1 - done_t) * self.gamma * max_next_q

            loss = self.loss_fn(q_val, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except Exception as e:
            # Si falla algo más, lo atrapamos aquí
            print(f"❌ Error crítico en learn(): {e}")
            raise e  # Lanzamos el error igual para que pare el programa

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
