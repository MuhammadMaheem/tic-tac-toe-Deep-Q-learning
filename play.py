

# --- file: play.py ---
"""
Play script for human vs AI Tic-Tac-Toe.

Loads the trained DQN model and provides a command-line interface
for playing against the AI agent.
"""

import torch
from game_logic import TicTacToe, render_board, smart_logic
from train import DQNAgent

MODEL_PATH = 'dqn_model.pth'


def human_move(env):
    """
    Get a valid move from the human player.

    Prompts for input, validates the move, and handles quit commands.

    Parameters
    ----------
    env : TicTacToe
        Current game environment.

    Returns
    -------
    int or None
        Valid action index (0-8) or None if user quits.
    """
    avail = env.available_actions()
    while True:
        try:
            user = input('Your move (1-9): ').strip()
            if user.lower() in ('q', 'quit', 'exit'):
                print('Exiting.')
                return None
            pos = int(user)
            if pos < 1 or pos > 9:
                print('Enter number 1-9.')
                continue
            action = pos - 1
            if action not in avail:
                print('Position taken or invalid. Choose another.')
                continue
            return action
        except ValueError:
            print('Enter a valid integer 1-9.')


def play():
    """
    Main game loop for human vs AI Tic-Tac-Toe.

    Loads the trained model, lets user choose who starts,
    and alternates turns until game ends.
    """
    device = 'cpu'
    env = TicTacToe()
    agent = DQNAgent(device=device)
    try:
        agent.load(MODEL_PATH)
        print(f'Loaded model from {MODEL_PATH}')
    except Exception as e:
        print('Could not load model, playing with untrained network (behaviour may be poor).')

    # Let human choose who starts
    while True:
        first = input('Who starts? (1 = You, 2 = AI) [default 1]: ').strip()
        if first == '' or first == '1':
            player = -1  # human is -1, will move first
            break
        if first == '2':
            player = 1
            break
        print('Choose 1 or 2')

    state = env.reset()

    while True:
        print('\n' + render_board(env) + '\n')

        if player == -1:
            action = human_move(env)
            if action is None:
                return
        else:
            # Use agent: set eps=0 to be greedy
            agent.eps = 0.0
            action = agent.choose_action(state, env, player)
            print(f'AI chooses: {action + 1}')

        env.make_move(action, player)

        if env.current_winner:
            print('\n' + render_board(env) + '\n')
            winner = 'AI' if env.current_winner == 1 else 'You'
            print(f'Winner: {winner}')
            break

        if env.is_draw():
            print('\n' + render_board(env) + '\n')
            print('Draw!')
            break

        state = env.get_state()
        player *= -1

    print('Game over.')


if __name__ == '__main__':
    play()
