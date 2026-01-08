
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

    def __init__(self, capacity=20000):
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

    def __init__(self, lr=1e-3, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.9995, device=None):
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
            Minimum epsilon value. Default is 0.05.
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


def make_random_opponent_move(env, player):
    """
    Make a random or smart opponent move.
    
    Parameters
    ----------
    env : TicTacToe
        Game environment.
    player : int
        Opponent player (should be -1 for opponent).
        
    Returns
    -------
    int or None
        Action taken, or None if no moves available.
    """
    # 70% smart logic, 30% random to add variety
    if random.random() < 0.7:
        smart = smart_logic(env, player)
        if smart is not None:
            return smart
    
    available = env.available_actions()
    if available:
        return random.choice(available)
    return None


def randomize_starting_position(env):
    """
    Initialize board with 0-3 random moves to explore different game states.
    
    Parameters
    ----------
    env : TicTacToe
        Game environment to randomize.
        
    Returns
    -------
    int
        Next player to move (1 or -1).
    """
    # 30% chance to start from random position
    if random.random() < 0.3:
        num_moves = random.randint(1, 3)  # 1-3 random moves
        player = 1
        for _ in range(num_moves):
            available = env.available_actions()
            if not available or env.current_winner or env.is_draw():
                break
            action = random.choice(available)
            env.make_move(action, player)
            player *= -1
        return player
    return 1  # Default: AI starts


def train(episodes=20000, batch_size=64):
    """
    Train the DQN agent through self-play and opponent play.

    The agent plays against itself and various opponents, collecting experiences
    from different starting positions and learning both offensive and defensive
    strategies. Uses epsilon decay for exploration-exploitation balance.

    Parameters
    ----------
    episodes : int, optional
        Number of training episodes. Default is 20000.
    batch_size : int, optional
        Batch size for training updates. Default is 64.
    """
    env = TicTacToe()
    agent = DQNAgent()

    wins = 0
    losses = 0
    draws = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        
        # Randomize starting position to explore all game states
        current_player = randomize_starting_position(env)
        state = env.get_state()
        
        # Randomize who the AI plays as (1 or -1) to learn both sides
        ai_player = random.choice([1, -1])
        
        # Choose opponent type: 60% self-play, 40% smart opponent
        use_opponent = random.random() < 0.4
        
        done = False
        player = current_player
        
        while not done:
            # AI's turn
            if player == ai_player:
                action = agent.choose_action(state, env, player)
                env.make_move(action, player)
                
                # Calculate reward
                reward = 0.0
                if env.current_winner == ai_player:
                    reward = 1.0
                    wins += 1
                    done = True
                elif env.current_winner == -ai_player:
                    reward = -1.0  # Penalty for losing
                    losses += 1
                    done = True
                elif env.is_draw():
                    reward = 0.5  # Draw is okay
                    draws += 1
                    done = True
                
                next_state = env.get_state()
                agent.push_memory((state, action, reward, next_state, done))
                
                # Train on batch
                agent.train_step(batch_size)
                
                state = next_state
            else:
                # Opponent's turn
                if use_opponent:
                    action = make_random_opponent_move(env, player)
                else:
                    # Self-play: AI plays both sides
                    action = agent.choose_action(state, env, player)
                
                if action is not None:
                    env.make_move(action, player)
                    
                    # Check game end from opponent's perspective
                    if env.current_winner == -ai_player:
                        reward = -1.0
                        losses += 1
                        done = True
                    elif env.current_winner == ai_player:
                        reward = 1.0
                        wins += 1
                        done = True
                    elif env.is_draw():
                        reward = 0.5
                        draws += 1
                        done = True
                    
                    state = env.get_state()
            
            player *= -1

        # Decay epsilon
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)

        if ep % 500 == 0:
            win_rate = wins / 500 * 100 if wins + losses + draws > 0 else 0
            print(f"Episode {ep}\tMemory:{len(agent.memory)}\tEps:{agent.eps:.4f}\tW/L/D: {wins}/{losses}/{draws} ({win_rate:.1f}% wins)")
            wins = losses = draws = 0

    agent.save()
    print(f"Training finished. Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    train()
