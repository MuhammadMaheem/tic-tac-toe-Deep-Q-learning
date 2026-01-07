
# --- file: train.py ---
"""
Training script for the DQN agent.

This module implements the Deep Q-Network (DQN) agent for Tic-Tac-Toe,
including the neural network architecture, experience replay memory,
and the training loop for self-play reinforcement learning.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from game_logic import TicTacToe, smart_logic

MODEL_PATH = 'dqn_model.pth'


class DQN(nn.Module):
    """
    Deep Q-Network for Tic-Tac-Toe.

    A simple feedforward neural network that approximates the Q-function.
    Takes board state as input and outputs Q-values for each action.
    """

    def __init__(self):
        """
        Initialize the DQN network.

        Architecture: Linear(9, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 9)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 9)

        Returns
        -------
        torch.Tensor
            Q-values for each of the 9 actions, shape (batch_size, 9)
        """
        return self.net(x)


class ReplayMemory:
    """
    Experience replay memory buffer.

    Stores past experiences for training stability and decorrelation.
    """

    def __init__(self, capacity=100000):
        """
        Initialize the replay memory.

        Parameters
        ----------
        capacity : int, optional
            Maximum number of experiences to store. Default is 50000.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        """
        Add a transition to the memory.

        Parameters
        ----------
        transition : tuple
            Experience tuple (state, action, reward, next_state, done)
        """
        self.memory.append(transition)

    def sample(self, n):
        """
        Sample a batch of experiences.

        Parameters
        ----------
        n : int
            Number of experiences to sample.

        Returns
        -------
        list
            List of n randomly sampled transitions.
        """
        return random.sample(self.memory, n)

    def __len__(self):
        """
        Get the current number of stored experiences.

        Returns
        -------
        int
            Number of experiences in memory.
        """
        return len(self.memory)


class DQNAgent:
    """
    Deep Q-Learning agent for Tic-Tac-Toe.

    Implements epsilon-greedy action selection, experience replay,
    and Q-learning updates.
    """

    def __init__(self, lr=1e-3, gamma=0.99, eps=1.0, eps_min=0.1, eps_decay=0.9995, device=None):
        """
        Initialize the DQN agent.

        Parameters
        ----------
        lr : float, optional
            Learning rate for optimizer. Default is 1e-3.
        gamma : float, optional
            Discount factor for future rewards. Default is 0.99.
        eps : float, optional
            Initial epsilon for exploration. Default is 1.0.
        eps_min : float, optional
            Minimum epsilon value. Default is 0.1.
        eps_decay : float, optional
            Epsilon decay factor per episode. Default is 0.9995.
        device : str, optional
            Device for PyTorch tensors ('cpu', 'cuda', or None for auto-detect). Default is None.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.memory = ReplayMemory()

    def choose_action(self, state, env, player):
        """
        Select an action using epsilon-greedy policy with smart logic.

        First tries rule-based smart moves, then epsilon-greedy with DQN.

        Parameters
        ----------
        state : np.ndarray
            Current board state (9 floats).
        env : TicTacToe
            Game environment.
        player : int
            Current player (1 or -1).

        Returns
        -------
        int
            Selected action index (0-8).
        """
        # Rule-based smart move
        smart = smart_logic(env, player)
        if smart is not None:
            return smart

        # Epsilon-greedy
        if random.random() < self.eps:
            return random.choice(env.available_actions())

        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(s).cpu().numpy().flatten()
        masked = np.full(9, -np.inf)
        for a in env.available_actions():
            masked[a] = q[a]
        return int(np.argmax(masked))

    def push_memory(self, transition):
        """
        Store an experience in replay memory.

        Parameters
        ----------
        transition : tuple
            Experience tuple (state, action, reward, next_state, done)
        """
        self.memory.push(transition)

    def train_step(self, batch_size=64):
        """
        Perform one training step on a batch of experiences.

        Parameters
        ----------
        batch_size : int, optional
            Number of experiences to sample. Default is 64.

        Returns
        -------
        float or None
            Loss value if training occurred, None if insufficient samples.
        """
        if len(self.memory) < batch_size:
            return None
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path=MODEL_PATH):
        """
        Save the model state dictionary to file.

        Parameters
        ----------
        path : str, optional
            File path to save the model. Default is MODEL_PATH.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path=MODEL_PATH):
        """
        Load the model state dictionary from file.

        Parameters
        ----------
        path : str, optional
            File path to load the model from. Default is MODEL_PATH.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def train(episodes=20000, batch_size=64):
    """
    Train the DQN agent through self-play.

    The agent plays against itself, collecting experiences and updating
    the network. Uses epsilon decay for exploration-exploitation balance.

    Parameters
    ----------
    episodes : int, optional
        Number of training episodes. Default is 20000.
    batch_size : int, optional
        Batch size for training updates. Default is 64.
    """
    env = TicTacToe()
    agent = DQNAgent()

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        player = 1  # AI starts
        while not done:
            action = agent.choose_action(state, env, player)
            env.make_move(action, player)

            reward = 0.0
            if env.current_winner == player:
                reward = 1.0
                done = True
            elif env.is_draw():
                reward = 0.3
                done = True

            next_state = env.get_state()
            agent.push_memory((state, action, reward, next_state, done))

            # Train on batch
            agent.train_step(batch_size)

            state = next_state
            player *= -1

        # Decay epsilon
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)

        if ep % 500 == 0:
            print(f"Episode {ep}\tMemory:{len(agent.memory)}\tEps:{agent.eps:.4f}")

    agent.save()
    print(f"Training finished. Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    train()
