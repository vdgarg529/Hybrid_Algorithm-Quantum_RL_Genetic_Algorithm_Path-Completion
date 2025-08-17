# === rl/dqn_agent.py ===
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
from torch.nn import functional as F
from typing import Deque, Dict, List, Tuple, Union


@dataclass
class Experience:
    obs: Union[Dict[str, np.ndarray], np.ndarray]
    actions: List[int]
    reward: float
    next_obs: Union[Dict[str, np.ndarray], np.ndarray]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, obs_mode: str, grid_size: int, action_dim: int):
        super().__init__()
        self.obs_mode = obs_mode
        self.action_dim = action_dim
        
        if obs_mode == "cnn":
            # CNN for image processing
            self.cnn = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate CNN output size
            with torch.no_grad():
                dummy = torch.zeros(1, 5, grid_size, grid_size)
                cnn_out_size = self.cnn(dummy).shape[1]
            
            # Combined features
            self.fc = nn.Sequential(
                nn.Linear(cnn_out_size + 8, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
        else:  # Tabular
            self.fc = nn.Sequential(
                nn.Linear(5 * grid_size**2 + 8, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
    
    def forward(self, obs):
        if self.obs_mode == "cnn":
            image = obs["image"]
            vector = obs["vector"]
            cnn_features = self.cnn(image)
            combined = torch.cat([cnn_features, vector], dim=1)
            return self.fc(combined)
        else:
            return self.fc(obs["flat"])


class DQNAgent:
    def __init__(
        self,
        obs_mode: str,
        grid_size: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_mode = obs_mode
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.steps = 0
        
        # Networks
        self.policy_net = DQN(obs_mode, grid_size, action_dim).to(self.device)
        self.target_net = DQN(obs_mode, grid_size, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def act(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> List[int]:
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if random.random() < self.epsilon:
            return [
                random.randint(0, 3),
                random.randint(0, 3)
            ]
        
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs)
            q_values = self.policy_net(obs_tensor).cpu().numpy().flatten()
        
        # Split Q-values for two UAVs
        action0 = np.argmax(q_values[:4])
        action1 = np.argmax(q_values[4:8])
        return [action0, action1]
    
    def store_experience(self, obs, actions, reward, next_obs, done):
        exp = Experience(obs, actions, reward, next_obs, done)
        self.replay_buffer.push(exp)
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        obs_batch = self._batch_to_tensor(batch)
        
        # Compute Q-values
        current_q = self.policy_net(obs_batch["obs"])
        next_q = self.target_net(obs_batch["next_obs"]).max(1, keepdim=True)[0]
        
        # Compute target Q-values
        target_q = current_q.clone()
        for i, exp in enumerate(batch):
            # Reward for both UAVs combined
            target = exp.reward
            if not exp.done:
                target += self.gamma * next_q[i].item()
            
            # Update Q-values for both actions
            target_q[i, exp.actions[0]] = target
            target_q[i, exp.actions[1] + 4] = target  # Second UAV actions offset by 4
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target_network()
    
    def _obs_to_tensor(self, obs):
        if self.obs_mode == "cnn":
            return {
                "image": torch.tensor(obs["image"], dtype=torch.float32)
                    .unsqueeze(0).to(self.device),
                "vector": torch.tensor(obs["vector"], dtype=torch.float32)
                    .unsqueeze(0).to(self.device)
            }
        else:
            return {
                "flat": torch.tensor(obs, dtype=torch.float32)
                    .unsqueeze(0).to(self.device)
            }
    
    def _batch_to_tensor(self, batch):
        if self.obs_mode == "cnn":
            obs_images = []
            obs_vectors = []
            next_obs_images = []
            next_obs_vectors = []
            
            for exp in batch:
                obs_images.append(exp.obs["image"])
                obs_vectors.append(exp.obs["vector"])
                next_obs_images.append(exp.next_obs["image"])
                next_obs_vectors.append(exp.next_obs["vector"])
            
            return {
                "obs": {
                    "image": torch.stack([torch.tensor(img, dtype=torch.float32) for img in obs_images]).to(self.device),
                    "vector": torch.stack([torch.tensor(vec, dtype=torch.float32) for vec in obs_vectors]).to(self.device)
                },
                "next_obs": {
                    "image": torch.stack([torch.tensor(img, dtype=torch.float32) for img in next_obs_images]).to(self.device),
                    "vector": torch.stack([torch.tensor(vec, dtype=torch.float32) for vec in next_obs_vectors]).to(self.device)
                }
            }
        else:
            obs_list = [exp.obs for exp in batch]
            next_obs_list = [exp.next_obs for exp in batch]
            return {
                "obs": torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in obs_list]).to(self.device),
                "next_obs": torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in next_obs_list]).to(self.device)
            }
    
    def _soft_update_target_network(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    def save(self, path):
        """Save policy network weights"""
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        """Load policy network weights"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()