import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from agent.model import DQN
from agent.replay_buffer import ReplayBuffer
from config import *

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(REPLAY_CAPACITY)
        
        self.epsilon = EPSILON_START
        self.steps_done = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Next Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * next_q * (1 - dones))
            
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
            
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
