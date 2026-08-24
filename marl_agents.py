import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QLearningAgent:
    """
    A classical Tabular Q-Learning Agent.
    Discretizes the continuous state space into a multi-dimensional grid
    and uses a Q-table to learn optimal actions.
    """
    def __init__(self, action_size=3, lr=0.1, gamma=0.95):
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.q_table = {}  # Keys: (discrete_state_tuple), Values: np.zeros(action_size)

    def discretize_state(self, state):
        """
        Converts continuous state [t_norm, SOC, price_norm, steps_left_norm, target_soc]
        into a discrete tuple key: (t, soc_bin, price_bin, steps_left_bin)
        """
        t = int(round(state[0] * 24)) % 24
        
        # SOC discretised into 10 bins (0 to 9)
        soc_bin = int(np.clip(state[1] * 10, 0, 9))
        
        # Price discretised into 5 bins (0 to 4)
        price_bin = int(np.clip(state[2] * 5, 0, 4))
        
        # Connection steps remaining discretised into 6 categories
        steps_left = int(round(state[3] * 24))
        if steps_left == 0:
            steps_bin = 0
        elif steps_left <= 1:
            steps_bin = 1
        elif steps_left <= 2:
            steps_bin = 2
        elif steps_left <= 4:
            steps_bin = 3
        elif steps_left <= 8:
            steps_bin = 4
        else:
            steps_bin = 5
            
        return (t, soc_bin, price_bin, steps_bin)

    def get_q_values(self, d_state):
        if d_state not in self.q_table:
            self.q_table[d_state] = np.zeros(self.action_size)
        return self.q_table[d_state]

    def choose_action(self, state, epsilon=0.1):
        d_state = self.discretize_state(state)
        q_vals = self.get_q_values(d_state)
        
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return int(np.argmax(q_vals))

    def learn(self, state, action, reward, next_state, done):
        d_state = self.discretize_state(state)
        d_next_state = self.discretize_state(next_state)
        
        q_vals = self.get_q_values(d_state)
        next_q_vals = self.get_q_values(d_next_state)
        
        max_next_q = 0.0 if done else np.max(next_q_vals)
        
        # Bellman Equation update
        td_target = reward + self.gamma * max_next_q
        q_vals[action] += self.lr * (td_target - q_vals[action])


class QNetwork(nn.Module):
    """PyTorch Deep Q-Network implementation."""
    def __init__(self, state_size=5, action_size=3):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    """
    A Deep Q-Network Agent.
    Handles continuous input states using a multi-layer perceptron neural network.
    Uses Experience Replay and Target Network structures.
    """
    def __init__(self, state_size=5, action_size=3, lr=0.001, gamma=0.95, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Double Deep Q-Network configurations
        self.q_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.update_target_network() # Init target network weights to match q_net
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Experience Replay Memory
        self.memory = deque(maxlen=buffer_size)
        
    def update_target_network(self):
        """Hard copy weights from Q Network to Target Network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_t)
        self.q_net.train()
        return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """Samples a batch from memory and performs a gradient descent step."""
        if len(self.memory) < self.batch_size:
            return 0.0  # Return zero loss if buffer doesn't have enough samples
            
        # Sample a batch of transitions
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to PyTorch Tensors
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Get Q-values for current actions taken
        q_values = self.q_net(states_t).gather(1, actions_t)
        
        # Compute target Q-values using the Target Network (DQN style)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + (self.gamma * max_next_q * (1.0 - dones_t))
            
        # Compute Mean Squared Error Loss
        loss = self.criterion(q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
