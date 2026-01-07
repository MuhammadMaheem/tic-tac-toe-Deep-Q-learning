# Tic-Tac-Toe with Deep Q-Network (DQN) AI 🤖❌⭕

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reinforcement learning implementation of Tic-Tac-Toe where an AI agent learns optimal gameplay using Deep Q-Network (DQN). Train the AI through self-play and challenge it in an interactive command-line interface or through a modern web interface!

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies and Concepts](#technologies-and-concepts)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the AI](#training-the-ai)
  - [Playing Against the AI (CLI)](#playing-against-the-ai-cli)
  - [Playing Against the AI (Web)](#playing-against-the-ai-web)
- [Code Explanation](#code-explanation)
- [Definitions and Explanations](#definitions-and-explanations)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements a Tic-Tac-Toe game where an AI agent learns to play optimally using Deep Q-Network (DQN), a reinforcement learning algorithm. The AI is trained through self-play, improving its strategy over time. Users can then play against the trained AI via a command-line interface or through an interactive web application.

The project consists of several Python scripts that handle game logic, training, and gameplay, along with a Flask web server and modern HTML/CSS/JavaScript frontend. It uses PyTorch for the neural network implementation and NumPy for numerical operations.

### Demo

```
1 | 2 | 3
---------
4 | X | 6
---------
7 | 8 | 9

AI chooses: 5
Your move (1-9): 1

X | O | 3
---------
4 | X | 6
---------
7 | 8 | 9

...
```

## Features

- 🧠 **Reinforcement Learning**: AI learns through self-play using DQN
- 🎮 **Interactive Gameplay**: Play against the trained AI in CLI or web interface
- 🌐 **Web Interface**: Modern, responsive web application with real-time gameplay
- 📊 **Experience Replay**: Stable training with memory buffer
- 🔄 **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- 💾 **Model Persistence**: Save and load trained models
- 🏆 **Smart Logic**: Basic rule-based AI for faster initial learning
- 📈 **Score Tracking**: Track wins, losses, and draws in web interface

## Technologies and Concepts

### Programming Language
- **Python**: The entire project is written in Python, a high-level, interpreted programming language known for its simplicity and extensive libraries for scientific computing and machine learning.

### Libraries

| Library | Purpose | Why Used | How It Works |
|---------|---------|----------|--------------|
| **NumPy** | Scientific computing, array operations | Represent Tic-Tac-Toe board as 3x3 array | Board initialized as `np.zeros((3,3), dtype=int)`; 0=empty, 1=AI(X), -1=Human(O) |
| **PyTorch** | Machine learning, neural networks | Implement DQN, tensor ops, training/inference | Subclass `nn.Module` for DQN network; handles autograd for backprop |
| **Flask** | Web framework, API server | Provide REST API for web-based gameplay | Routes for new games, player moves, and game state; serves static files |

### Key Concepts

#### Reinforcement Learning (RL)
Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and aims to maximize cumulative rewards over time.

- **Why used**: RL is ideal for game-playing AI because it allows the agent to learn optimal strategies through trial and error, without explicit programming of game rules.
- **How it works in this project**: The AI agent plays Tic-Tac-Toe games against itself. It receives rewards for winning (+1), draws (+0.3), and losses (0). Over many episodes, it learns to choose actions that maximize future rewards.

#### Deep Q-Network (DQN)
DQN is a deep learning variant of Q-learning, a model-free RL algorithm. It uses a neural network to approximate the Q-function, which estimates the expected future rewards for each action in a given state.

- **Why used**: DQN can handle complex state spaces that traditional Q-learning cannot. Tic-Tac-Toe's state space (3^9 = 19683 possible states) is manageable but benefits from the generalization capabilities of neural networks.
- **How it works**: The DQN takes the current board state as input (flattened 9-element array) and outputs Q-values for each of the 9 possible actions. The agent selects actions using an epsilon-greedy policy during training.

#### Experience Replay
A technique where the agent stores past experiences (state, action, reward, next state) in a buffer and samples random batches for training. This breaks correlations between consecutive experiences and improves learning stability.

- **Why used**: Experience replay helps stabilize training by reducing variance and allowing the agent to learn from diverse experiences multiple times.
- **How it works**: Experiences are stored in a deque (double-ended queue) with a maximum capacity. During training, random batches are sampled to update the network.

#### Epsilon-Greedy Policy
A policy that selects the best action with probability (1 - ε) and a random action with probability ε. The ε value decays over time to shift from exploration to exploitation.

- **Why used**: This balances exploration (trying new actions) and exploitation (using known good actions) during training.
- **How it works**: Initially, ε = 1.0 (mostly random actions). It decays to ε_min = 0.1 over episodes, allowing the agent to explore early and exploit learned knowledge later.

#### Tic-Tac-Toe Game Logic
The rules and mechanics of the Tic-Tac-Toe game, including board representation, move validation, win checking, and draw detection.

- **Why used**: Provides the environment for the RL agent to interact with.
- **How it works**: The board is a 3x3 grid. Players alternate placing X's and O's. A player wins by getting three in a row (horizontally, vertically, or diagonally). If the board fills without a winner, it's a draw.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tic-tac-toe.git
   cd tic-tac-toe
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the AI

Run the training script to train the DQN agent:

```bash
python train.py
```

- The script trains for 20,000 episodes by default
- Progress is printed every 500 episodes
- The trained model is saved as `dqn_model.pth`

### Playing Against the AI (CLI)

After training, play against the AI:

```bash
python play.py
```

- Choose who starts (you or AI)
- Enter moves as numbers 1-9 corresponding to board positions
- The AI will respond with its moves
- Game continues until someone wins or it's a draw

### Playing Against the AI (Web)

Launch the web server for an interactive web-based experience:

```bash
python server.py
```

- Open your browser and navigate to `http://localhost:8000`
- Click "New Game" to start (choose if you or AI goes first)
- Click on board cells to make your moves
- The AI responds automatically
- Track your wins, losses, and draws
- Modern, responsive interface with animations

## Code Overview

The project is organized into several Python files and a web frontend:

- `game_logic.py`: Implements the TicTacToe environment, including board management, move validation, win checking, and rendering.
- `train.py`: Contains the DQN neural network, replay memory, agent class, and training loop for self-play.
- `play.py`: Provides an interactive CLI for playing against the trained AI.
- `server.py`: Flask web server that provides REST API endpoints for web-based gameplay.
- `static/index.html`: Main HTML page for the web interface.
- `static/style.css`: CSS styles for the modern, animated web interface.
- `static/script.js`: JavaScript for handling user interactions and API calls.

For detailed code explanations and inline comments, refer to the source files directly.

## Definitions and Explanations with Examples

### State
In RL, a state represents the current situation of the environment. In this project, the state is the flattened 9-element array representing the Tic-Tac-Toe board.

**Example**: An empty board is `[0, 0, 0, 0, 0, 0, 0, 0, 0]`. After X plays in the center: `[0, 0, 0, 0, 1, 0, 0, 0, 0]`.

### Action
An action is a move the agent can take. In Tic-Tac-Toe, actions are the 9 possible board positions (0-8).

**Example**: Action 4 places a mark in the center of the board.

### Reward
A scalar value given to the agent after taking an action. It indicates how good the action was.

**Example**: +1 for winning, +0.3 for draw, 0 for loss or non-terminal moves.

### Episode
A complete sequence of states, actions, and rewards from start to end of a game.

**Example**: One full Tic-Tac-Toe game from empty board to win/draw.

### Q-Value
The expected future reward for taking a specific action in a specific state.

**Example**: Q(state, action=4) = 0.8 means taking action 4 in that state leads to expected reward of 0.8.

### Neural Network Forward Pass
The process of feeding input through the network to get output.

**Example**: Input `[0,0,0,0,1,0,0,0,0]` → Network → Output `[0.1, 0.2, 0.3, 0.4, 0.8, 0.6, 0.7, 0.9, 0.5]` (Q-values for each action).

### Backpropagation
The algorithm for computing gradients and updating network weights.

**Example**: After computing loss, gradients flow backward through the network, adjusting weights to reduce future losses.

### Loss Function
A measure of how well the network's predictions match the targets.

**Example**: MSE Loss = mean((predicted_Q - target_Q)^2) across the batch.

This README provides a comprehensive overview of the Tic-Tac-Toe DQN project, covering all aspects of the code, concepts, and workflow in detail.

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

⭐ If you find this project helpful, please give it a star!