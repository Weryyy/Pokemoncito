import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
from src.models.dqn_combat import CombatDQN


class ReplayBuffer:
    """Experience replay buffer for better learning stability"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class TacticianAgent:
    def __init__(self, input_dim, n_actions, lr=1e-3):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions

        # DQN con Target Network
        self.policy_net = CombatDQN(input_dim, n_actions).to(self.device)
        self.target_net = CombatDQN(input_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Experience Replay
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.target_update_frequency = 100
        self.learn_step_counter = 0

        # Hiperparámetros mejorados
        self.epsilon = 1.0
        self.epsilon_min = 0.1
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
            # Almacenar en buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Solo entrenar si tenemos suficientes muestras
            if len(self.replay_buffer) < self.batch_size:
                return
            
            # Muestrear batch del replay buffer con manejo de errores
            try:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            except (ValueError, IndexError) as e:
                # Si falla el muestreo, esperar más samples
                return
            
            states_t = torch.FloatTensor(states).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)

            # Q-values actuales
            q_values = self.policy_net(states_t)
            q_val = q_values.gather(1, actions_t).squeeze(1)

            # Q-values objetivo usando target network
            with torch.no_grad():
                next_q_values = self.target_net(next_states_t)
                max_next_q = next_q_values.max(1)[0]
                target = rewards_t + (1 - dones_t) * self.gamma * max_next_q

            loss = self.loss_fn(q_val, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Actualizar target network periódicamente
            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        except Exception as e:
            print(f"❌ Error crítico en learn(): {e}")
            raise e

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
