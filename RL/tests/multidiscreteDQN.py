import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gym.spaces import MultiDiscrete


class MultiDiscreteDQN:
    def __init__(self, state_space, action_space, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self.create_model().to(self.device)
        self.target_net = self.create_model().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def create_model(self):
        class MultiDiscreteDQNet(nn.Module):
            def __init__(self, input_shape, output_shape):
                super(MultiDiscreteDQNet, self).__init__()
                self.fc1 = nn.Linear(input_shape[0], 24)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, np.prod(output_shape))
                self.output_shape = output_shape

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x.reshape(-1, *self.output_shape)

        return MultiDiscreteDQNet(self.state_space.shape, self.action_space.shape)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).cpu().numpy().reshape(-1)

        return action

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
