import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
import CityFlowRL

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
                self.fc1 = nn.Linear(int(np.prod(input_shape)), 24)  # the input shape here is [8*3]--> 24 and the states per batch shape is [64,8,3]
                self.fc2 = nn.Linear(24, 64)
                self.fc3 = nn.Linear(64, 4)
                # self.fc3s = nn.ModuleList([nn.Linear(64, n) for n in output_shape])
                # self.fc3 = nn.Linear(64,int(np.prod(output_shape)))
                # self.output_shape = output_shape

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                # xs = [fc(x) for fc in self.fc3s]
                x = self.fc3(x)
                if len(x) == 0:
                    return torch.empty((x.size(0),) + (5, 1), device=x.device)
                else:
                    return x
                    # return torch.cat(x, dim=-1)

                # x = x.view(x.size(0),-1)
                # x = torch.relu(self.fc1(x))
                # x = torch.relu(self.fc2(x))
                # xs = [fc(x) for fc in self.fc3s]
                # if len(xs) == 0:
                #     return torch.empty((x.size(0), 0) + self.output_shape, device=x.device)
                # else:
                #     return torch.stack(xs, dim=-1)

                # #x = self.fc3(x)
                # return torch.stack(xs, dim=-1)
                # #return x.reshape(-1, *self.output_shape)

        return MultiDiscreteDQNet(self.state_space.shape, self.action_space.shape)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                print(state)
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                print("State shape: ", state.shape)
                print("Q-values shape: ", q_values.shape)
                action = int(q_values.argmax(dim=-1).cpu().numpy().reshape(-1))

        return action

    def remember(self, state, action, reward, next_state, done):
        print(state, action, reward, next_state, done)
        self.buffer.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        states = states.view(self.batch_size, 1, -1)  # Reshape to (batch_size, 1, state_size)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # actions = actions.view(self.batch_size, 1) # Reshape to (batch_size, 1)
        print(states.shape)
        print(actions.shape)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        policy_output = self.policy_net(states)
        print("policy_output shape:", policy_output.shape)
        print("actions shape:", actions.shape)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # def learn(self):
    #     if len(self.buffer) < self.batch_size:
    #         return

    #     batch = random.sample(self.buffer, self.batch_size)
    #     states, actions, rewards, next_states, dones = zip(*batch)

    #     states = torch.FloatTensor(states).to(self.device)
    #     actions = torch.LongTensor(actions).to(self.device)
    #     rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
    #     next_states = torch.FloatTensor(next_states).to(self.device)
    #     dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

    #     q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    #     next_q_values = self.target_net(next_states).max(1)[0].detach()
    #     target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

    #     loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

    #     self.optimizer.zero_grad()
    #     loss.backward()


if __name__ == "__main__":
    models_dir = "../models/"
    env_kwargs = {'config': "1x1_config", 'steps_per_episode': 100, 'steps_per_action': 30}
    env = gym.make('CityFlowRL-v0', **env_kwargs)
    env.set_save_replay(False)
    # print(env._get_state())
    # print(type(env._get_state()))
    # print(env.observation_space.sample())
    # print(type(env.observation_space.sample()))
    # print(len(env.observation_space.sample()))
    # action = env.action_space.sample()
    # print(action)
    # print(env.observation_space.shape)
    # print(env._get_state())
    # print(env.step(action))
    obs = env.observation_space
    action = env.action_space
    model = MultiDiscreteDQN(obs, action)
    # state = env.reset()
    # action = model.act(state)   #_states
    # next_state,reward,done,info= env.step(action)
    # model.remember(state, action, reward, next_state, done)
    total_episodes = 100
    model.create_model()
    # model.compile(loss='mse', optimizer=Adam(learning_rate = 0.001))
    # model.fit()

    for episode in range(1, total_episodes + 1):
        is_done = False
        score = 0
        state = env.reset()
        print("hi reeeeeeeeeeeessssssssssseeeeeeeeeeettttttttt")
        # print(done)
        while is_done == False:
            action = model.act(state)
            print("action:", action)
            print(type(action))
            next_state, reward, is_done, info = env.step(action)
            model.remember(state, action, reward, next_state, is_done)

            model.learn()
            state = next_state
        model.epsilon = max(model.epsilon_min, model.epsilon * model.epsilon_decay)
        # print(done)

    # Test the DQN for 10 episodes
    env.set_save_replay(True)
    num_test_episodes = 10
    for episode in range(num_test_episodes):
        state = env.reset()
        is_done = False
        total_reward = 0
        while is_done == False:
            # Choose the action with the highest Q-value
            action = model.act(state)
            # Take the action and observe the next state and reward
            next_state, reward, is_done, info = env.step(action)

            # Update the current state
            state = next_state
            total_reward += reward

        # Print the total reward earned in the episode
        print(f"Episode {episode + 1}: Total reward = {total_reward}")
