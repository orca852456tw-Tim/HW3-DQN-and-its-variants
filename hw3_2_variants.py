import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from Gridworld import Gridworld
import os

# Hyperparameters
LR = 1e-3
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.1
MEMORY_SIZE = 2000
BATCH_SIZE = 150
TARGET_UPDATE = 500
EPISODES = 1500

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def get_state(env):
    state = env.board.render_np().astype(np.float32)
    return state.flatten()

# --- Models ---
class BasicDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(BasicDQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# --- Training Loop ---
def train_agent(variant_name="Basic"):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    if variant_name == "Dueling":
        model = DuelingDQN(input_dim=64, output_dim=4)
        target_model = DuelingDQN(input_dim=64, output_dim=4)
    else:
        model = BasicDQN(input_dim=64, output_dim=4)
        target_model = BasicDQN(input_dim=64, output_dim=4)
        
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_START - EPSILON_END) / EPISODES
    
    total_steps = 0
    rewards_history = []
    losses_history = []
    
    for episode in range(EPISODES):
        env = Gridworld(size=4, mode='player')
        state = get_state(env)
        done = False
        episode_reward = 0
        step_count = 0
        episode_loss = []
        
        while not done and step_count < 50:
            if random.random() < epsilon:
                action_idx = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state).unsqueeze(0))
                    action_idx = q_values.argmax().item()
            
            action = action_set[action_idx]
            env.makeMove(action)
            reward = env.reward()
            next_state = get_state(env)
            
            done = reward == 10 or reward == -10
            
            memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            step_count += 1
            
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                states = torch.tensor(states)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)
                
                curr_q = model(states).gather(1, actions).squeeze()
                
                if variant_name == "Double":
                    # Double DQN
                    with torch.no_grad():
                        next_actions = model(next_states).argmax(1, keepdim=True)
                        next_q = target_model(next_states).gather(1, next_actions).squeeze()
                        target_q = rewards + GAMMA * next_q * (1 - dones)
                else:
                    # Basic & Dueling
                    with torch.no_grad():
                        next_q = target_model(next_states).max(1)[0]
                        target_q = rewards + GAMMA * next_q * (1 - dones)
                
                loss = criterion(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss.append(loss.item())
                
            if total_steps % TARGET_UPDATE == 0:
                target_model.load_state_dict(model.state_dict())
                
        epsilon = max(EPSILON_END, epsilon - epsilon_decay)
        rewards_history.append(episode_reward)
        if episode_loss:
            losses_history.append(np.mean(episode_loss))
        else:
            losses_history.append(0)
            
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f'{variant_name} | Episode {episode + 1}/{EPISODES}, Epsilon: {epsilon:.2f}, Avg Reward: {avg_reward:.2f}')

    return rewards_history, losses_history

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    print("Training Basic DQN...")
    r_basic, l_basic = train_agent("Basic")
    
    print("Training Double DQN...")
    r_double, l_double = train_agent("Double")
    
    print("Training Dueling DQN...")
    r_dueling, l_dueling = train_agent("Dueling")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Rewards plot
    plt.subplot(1, 2, 1)
    plt.plot(moving_average(r_basic), label='Basic DQN', alpha=0.8)
    plt.plot(moving_average(r_double), label='Double DQN', alpha=0.8)
    plt.plot(moving_average(r_dueling), label='Dueling DQN', alpha=0.8)
    plt.title('Reward (100-Episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(moving_average(l_basic), label='Basic DQN', alpha=0.8)
    plt.plot(moving_average(l_double), label='Double DQN', alpha=0.8)
    plt.plot(moving_average(l_dueling), label='Dueling DQN', alpha=0.8)
    plt.title('Training Loss (100-Episode Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dqn_comparison.png')
    print("Saved dqn_comparison.png")

if __name__ == "__main__":
    main()
