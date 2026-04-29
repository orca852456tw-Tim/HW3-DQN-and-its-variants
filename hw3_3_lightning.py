import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import random
from collections import deque
from Gridworld import Gridworld
from torch.utils.data import DataLoader, IterableDataset

# Hyperparameters
LR = 1e-3
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.1
MEMORY_SIZE = 2000
BATCH_SIZE = 150
TARGET_UPDATE = 500
EPISODES = 3000
MAX_STEPS = 50

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

class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size):
        self.buffer = buffer
        self.sample_size = sample_size
        
    def __iter__(self):
        for _ in range(self.sample_size):
            yield self.buffer.sample(1)

def get_state(env):
    state = env.board.render_np().astype(np.float32)
    return state.flatten()

class DuelingDQNLightning(pl.LightningModule):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQNLightning, self).__init__()
        self.save_hyperparameters()
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        
        # Networks
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_dim))
        
        self.target_feature_layer = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.target_value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.target_advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_dim))
        
        self.target_feature_layer.load_state_dict(self.feature_layer.state_dict())
        self.target_value_stream.load_state_dict(self.value_stream.state_dict())
        self.target_advantage_stream.load_state_dict(self.advantage_stream.state_dict())
        
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.env = Gridworld(size=4, mode='random')
        self.epsilon = EPSILON_START
        self.epsilon_decay = (EPSILON_START - EPSILON_END) / EPISODES
        
        self.total_steps = 0
        self.episode_reward = 0
        self.state = get_state(self.env)
        self.criterion = nn.MSELoss()
        
        # Populate buffer initially
        self._populate_buffer()

    def _populate_buffer(self):
        # Fill buffer with random actions initially to have at least BATCH_SIZE
        state = get_state(self.env)
        for _ in range(BATCH_SIZE):
            action_idx = random.randint(0, 3)
            self.env.makeMove(self.action_set[action_idx])
            reward = self.env.reward()
            next_state = get_state(self.env)
            done = reward == 10 or reward == -10
            self.memory.push(state, action_idx, reward, next_state, done)
            if done:
                self.env = Gridworld(size=4, mode='random')
                state = get_state(self.env)
            else:
                state = next_state
        self.state = state
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def target_forward(self, x):
        features = self.target_feature_layer(x)
        values = self.target_value_stream(features)
        advantages = self.target_advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def play_step(self):
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self(torch.tensor(self.state).unsqueeze(0).to(self.device))
                action_idx = q_values.argmax().item()
        
        self.env.makeMove(self.action_set[action_idx])
        reward = self.env.reward()
        next_state = get_state(self.env)
        done = reward == 10 or reward == -10
        
        self.memory.push(self.state, action_idx, reward, next_state, done)
        self.episode_reward += reward
        self.total_steps += 1
        
        if done or (self.total_steps % MAX_STEPS == 0):
            self.env = Gridworld(size=4, mode='random')
            self.state = get_state(self.env)
            self.epsilon = max(EPSILON_END, self.epsilon - self.epsilon_decay)
            self.log('episode_reward', self.episode_reward, prog_bar=True)
            self.episode_reward = 0
        else:
            self.state = next_state

        if self.total_steps % TARGET_UPDATE == 0:
            self.target_feature_layer.load_state_dict(self.feature_layer.state_dict())
            self.target_value_stream.load_state_dict(self.value_stream.state_dict())
            self.target_advantage_stream.load_state_dict(self.advantage_stream.state_dict())

    def training_step(self, batch, batch_idx):
        self.play_step() # Collect experience
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        curr_q = self(states).gather(1, actions).squeeze()
        
        with torch.no_grad():
            next_q = self.target_forward(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
            
        loss = self.criterion(curr_q, target_q)
        self.log('train_loss', loss)
        self.log('val_loss', loss) # Mock val loss for ReduceLROnPlateau
        return loss

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # Custom gradient clipping as requested (max norm = 10.0)
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)

    def train_dataloader(self):
        dataset = RLDataset(self.memory, 100) # dummy size
        return DataLoader(dataset, batch_size=None)

def main():
    model = DuelingDQNLightning()
    trainer = pl.Trainer(max_epochs=EPISODES // 100, log_every_n_steps=10) # Using epochs as a proxy for episodes
    trainer.fit(model)
    
    torch.save(model.state_dict(), 'hw3_3_lightning_model.pth')
    print("Model saved to hw3_3_lightning_model.pth")

if __name__ == "__main__":
    main()
