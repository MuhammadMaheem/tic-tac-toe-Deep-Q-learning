"""
Flask server for Tic-Tac-Toe AI game.
Provides API endpoints for web-based gameplay.
"""

from flask import Flask, jsonify, request, send_from_directory
import torch
import numpy as np
from game_logic import TicTacToe, render_board
from train import DQNAgent
import os

app = Flask(__name__, static_folder='static', static_url_path='')

MODEL_PATH = 'dqn_model.pth'

# Global game state
game_env = TicTacToe()
agent = DQNAgent(device='cpu')

# Load the trained model
try:
    agent.load(MODEL_PATH)
    print(f'✓ Loaded model from {MODEL_PATH}')
except Exception as e:
    print(f'⚠ Could not load model: {e}')
    print('Using untrained network (behavior may be poor)')


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """
    Start a new game.
    
    Request JSON:
        {
            "ai_starts": boolean (optional, default: false)
        }
    
    Returns:
        {
            "board": list of 9 integers,
            "game_over": boolean,
            "winner": int or null,
            "is_draw": boolean,
            "message": string
        }
    """
    global game_env
    
    data = request.get_json() or {}
    ai_starts = data.get('ai_starts', False)
    
    game_env.reset()
    
    response = {
        'board': game_env.board.flatten().tolist(),
        'game_over': False,
        'winner': None,
        'is_draw': False,
        'message': 'New game started!'
    }
    
    # If AI starts, make the first move
    if ai_starts:
        state = game_env.get_state()
        agent.eps = 0.0  # Greedy mode
        action = agent.choose_action(state, game_env, player=1)
        game_env.make_move(action, player=1)
        
        response['board'] = game_env.board.flatten().tolist()
        response['message'] = f'AI moved to position {action + 1}'
    
    return jsonify(response)


@app.route('/api/player_move', methods=['POST'])
def player_move():
    """
    Process player's move and get AI response.
    
    Request JSON:
        {
            "position": int (0-8)
        }
    
    Returns:
        {
            "board": list of 9 integers,
            "game_over": boolean,
            "winner": int or null (-1 for player, 1 for AI),
            "is_draw": boolean,
            "message": string,
            "ai_position": int or null
        }
    """
    global game_env
    
    data = request.get_json()
    if not data or 'position' not in data:
        return jsonify({'error': 'Missing position parameter'}), 400
    
    position = data['position']
    
    # Validate position
    if not isinstance(position, int) or position < 0 or position > 8:
        return jsonify({'error': 'Invalid position. Must be 0-8'}), 400
    
    # Check if position is available
    if position not in game_env.available_actions():
        return jsonify({'error': 'Position already taken'}), 400
    
    # Player move (player = -1)
    game_env.make_move(position, player=-1)
    
    # Check if player won or draw
    if game_env.current_winner == -1:
        return jsonify({
            'board': game_env.board.flatten().tolist(),
            'game_over': True,
            'winner': -1,
            'is_draw': False,
            'message': 'You won! 🎉',
            'ai_position': None
        })
    
    if game_env.is_draw():
        return jsonify({
            'board': game_env.board.flatten().tolist(),
            'game_over': True,
            'winner': None,
            'is_draw': True,
            'message': "It's a draw!",
            'ai_position': None
        })
    
    # AI move (player = 1)
    state = game_env.get_state()
    agent.eps = 0.0  # Greedy mode
    ai_action = agent.choose_action(state, game_env, player=1)
    game_env.make_move(ai_action, player=1)
    
    # Check if AI won or draw
    message = f'AI moved to position {ai_action + 1}'
    game_over = False
    winner = None
    is_draw = False
    
    if game_env.current_winner == 1:
        message = 'AI won! Try again!'
        game_over = True
        winner = 1
    elif game_env.is_draw():
        message = "It's a draw!"
        game_over = True
        is_draw = True
    
    return jsonify({
        'board': game_env.board.flatten().tolist(),
        'game_over': game_over,
        'winner': winner,
        'is_draw': is_draw,
        'message': message,
        'ai_position': ai_action
    })


@app.route('/api/game_state', methods=['GET'])
def game_state():
    """
    Get current game state.
    
    Returns:
        {
            "board": list of 9 integers,
            "game_over": boolean,
            "winner": int or null,
            "is_draw": boolean,
            "available_actions": list of int
        }
    """
    global game_env
    
    return jsonify({
        'board': game_env.board.flatten().tolist(),
        'game_over': game_env.current_winner is not None or game_env.is_draw(),
        'winner': game_env.current_winner,
        'is_draw': game_env.is_draw(),
        'available_actions': game_env.available_actions()
    })


if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    print('='*50)
    print('🎮 Tic-Tac-Toe AI Server Starting...')
    print('='*50)
    print('Open your browser and navigate to:')
    print('  👉 http://localhost:8000')
    print('='*50)
    
    app.run(host='0.0.0.0', port=8000, debug=True)
