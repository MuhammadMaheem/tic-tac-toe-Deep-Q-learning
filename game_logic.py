# --- file: game_logic.py ---
"""
Game logic for Tic-Tac-Toe.

This module implements the core game mechanics for Tic-Tac-Toe, including
board management, move validation, win checking, and rendering. It serves
as the environment for reinforcement learning agents.
"""
import numpy as np


class TicTacToe:
    """
    Tic-Tac-Toe game environment.

    Manages the game state, including the board, current winner, and game logic.
    The board uses 0 for empty cells, 1 for AI (X), and -1 for Human (O).
    """

    def __init__(self):
        """
        Initialize the Tic-Tac-Toe game.

        Creates a 3x3 board filled with zeros and sets no current winner.
        """
        # 0 = empty, 1 = AI (X), -1 = Human (O)
        self.board = np.zeros((3, 3), dtype=int)
        self.current_winner = None

    def reset(self):
        """
        Reset the game to initial state.

        Returns
        -------
        np.ndarray
            Flattened board state as float32 array.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_winner = None
        return self.get_state()

    def get_state(self):
        """
        Get the current board state as a flattened array.

        Returns
        -------
        np.ndarray
            1D array of 9 float32 values representing the board state.
        """
        return self.board.flatten().astype(np.float32)

    def available_actions(self):
        """
        Get list of available actions (empty positions).

        Returns
        -------
        list of int
            Indices (0-8) of empty board positions.
        """
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def make_move(self, action, player):
        """
        Attempt to make a move on the board.

        Parameters
        ----------
        action : int
            Position index (0-8) to place the mark.
        player : int
            Player identifier (1 for AI, -1 for Human).

        Returns
        -------
        bool
            True if move was successful, False if position occupied.
        """
        x, y = divmod(action, 3)
        if self.board[x][y] != 0:
            return False
        self.board[x][y] = player
        if self.check_winner(player):
            self.current_winner = player
        return True

    def check_winner(self, p):
        """
        Check if a player has won the game.

        Parameters
        ----------
        p : int
            Player identifier to check for win.

        Returns
        -------
        bool
            True if player p has three in a row, False otherwise.
        """
        b = self.board
        # Check rows and columns
        for i in range(3):
            if all(b[i, :] == p) or all(b[:, i] == p):
                return True
        # Check diagonals
        if b[0, 0] == b[1, 1] == b[2, 2] == p:
            return True
        if b[0, 2] == b[1, 1] == b[2, 0] == p:
            return True
        return False

    def is_draw(self):
        """
        Check if the game is a draw.

        Returns
        -------
        bool
            True if board is full and no winner, False otherwise.
        """
        return np.all(self.board != 0) and self.current_winner is None


def check_win_flat(flat_board, p):
    """
    Check for a win condition on a flattened board.

    Parameters
    ----------
    flat_board : array-like
        1D array of 9 values representing the board.
    p : int
        Player identifier to check for win.

    Returns
    -------
    bool
        True if player p has three in a row, False otherwise.
    """
    b = np.array(flat_board).reshape(3, 3)
    # Check rows and columns
    for i in range(3):
        if all(b[i, :] == p) or all(b[:, i] == p):
            return True
    # Check diagonals
    if b[0, 0] == b[1, 1] == b[2, 2] == p:
        return True
    if b[0, 2] == b[1, 1] == b[2, 0] == p:
        return True
    return False


def smart_logic(env, player):
    """
    Implement basic AI logic to find winning or blocking moves.

    First attempts to find a winning move, then a blocking move to prevent
    opponent win. Returns None if no smart move found.

    Parameters
    ----------
    env : TicTacToe
        Game environment instance.
    player : int
        Current player identifier.

    Returns
    -------
    int or None
        Action index (0-8) for smart move, or None.
    """
    board = env.board.flatten()

    # Try to win
    for action in env.available_actions():
        temp = board.copy()
        temp[action] = player
        if check_win_flat(temp, player):
            return action

    # Try to block opponent
    opp = -player
    for action in env.available_actions():
        temp = board.copy()
        temp[action] = opp
        if check_win_flat(temp, opp):
            return action

    return None


def render_board(env):
    """
    Render the current board state as a string.

    Displays the board with X, O, or position numbers for empty cells.

    Parameters
    ----------
    env : TicTacToe
        Game environment instance.

    Returns
    -------
    str
        Formatted string representation of the board.
    """
    b = env.board.flatten()
    symbols = []
    for i, v in enumerate(b):
        if v == 1:
            symbols.append('X')
        elif v == -1:
            symbols.append('O')
        else:
            symbols.append(str(i + 1))
    rows = [" | ".join(symbols[i * 3:(i + 1) * 3]) for i in range(3)]
    sep = "\n---------\n"
    return sep.join(rows)

