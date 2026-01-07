// ==========================================
// GAME STATE
// ==========================================

let gameState = {
    board: Array(9).fill(0),
    gameOver: false,
    playerWins: 0,
    aiWins: 0,
    draws: 0
};

// ==========================================
// API CALLS
// ==========================================

async function apiCall(endpoint, data = null) {
    try {
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`/api/${endpoint}`, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'API request failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showMessage('❌ Error: ' + error.message, 'error');
        return null;
    }
}

// ==========================================
// GAME FUNCTIONS
// ==========================================

async function startNewGame(aiStarts = false) {
    // Disable buttons during request
    disableButtons(true);
    
    const result = await apiCall('new_game', { ai_starts: aiStarts });
    
    if (result) {
        gameState.board = result.board;
        gameState.gameOver = result.game_over;
        updateBoard();
        showMessage(result.message);
    }
    
    disableButtons(false);
}

async function makePlayerMove(position) {
    if (gameState.gameOver) {
        showMessage('Game over! Start a new game.', 'warning');
        return;
    }
    
    if (gameState.board[position] !== 0) {
        showMessage('Position already taken!', 'warning');
        return;
    }
    
    // Disable board during AI thinking
    disableBoard(true);
    
    const result = await apiCall('player_move', { position: position });
    
    if (result) {
        gameState.board = result.board;
        gameState.gameOver = result.game_over;
        
        updateBoard();
        showMessage(result.message);
        
        if (result.game_over) {
            handleGameOver(result);
        }
    }
    
    disableBoard(false);
}

function handleGameOver(result) {
    if (result.winner === -1) {
        // Player won
        gameState.playerWins++;
        updateScore();
        setTimeout(() => showWinnerModal('🎉', 'You Won!', 'Congratulations! You beat the AI!'), 500);
    } else if (result.winner === 1) {
        // AI won
        gameState.aiWins++;
        updateScore();
        setTimeout(() => showWinnerModal('🤖', 'AI Won!', 'The AI outsmarted you this time!'), 500);
    } else if (result.is_draw) {
        // Draw
        gameState.draws++;
        updateScore();
        setTimeout(() => showWinnerModal('🤝', "It's a Draw!", 'Well played! Nobody won this round.'), 500);
    }
    
    highlightWinningCells();
}

// ==========================================
// UI UPDATE FUNCTIONS
// ==========================================

function updateBoard() {
    const cells = document.querySelectorAll('.cell');
    
    cells.forEach((cell, index) => {
        const value = gameState.board[index];
        
        // Clear previous content and classes
        cell.textContent = '';
        cell.classList.remove('player-move', 'ai-move', 'taken', 'winning-cell');
        
        if (value === -1) {
            cell.textContent = 'O';
            cell.classList.add('player-move', 'taken');
        } else if (value === 1) {
            cell.textContent = 'X';
            cell.classList.add('ai-move', 'taken');
        }
    });
}

function showMessage(text, type = 'info') {
    const messageBox = document.getElementById('message');
    const messageText = document.getElementById('message-text');
    
    messageText.textContent = text;
    
    // Reset animation
    messageBox.style.animation = 'none';
    setTimeout(() => {
        messageBox.style.animation = '';
    }, 10);
}

function updateScore() {
    document.getElementById('player-wins').textContent = gameState.playerWins;
    document.getElementById('ai-wins').textContent = gameState.aiWins;
    document.getElementById('draws').textContent = gameState.draws;
}

function disableButtons(disabled) {
    document.getElementById('new-game-btn').disabled = disabled;
    document.getElementById('ai-start-btn').disabled = disabled;
}

function disableBoard(disabled) {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => {
        if (disabled) {
            cell.style.pointerEvents = 'none';
            cell.style.opacity = '0.7';
        } else {
            cell.style.pointerEvents = '';
            cell.style.opacity = '';
        }
    });
}

function highlightWinningCells() {
    const board = gameState.board;
    const winPatterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
        [0, 4, 8], [2, 4, 6]             // Diagonals
    ];
    
    for (const pattern of winPatterns) {
        const [a, b, c] = pattern;
        if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) {
            const cells = document.querySelectorAll('.cell');
            cells[a].classList.add('winning-cell');
            cells[b].classList.add('winning-cell');
            cells[c].classList.add('winning-cell');
            break;
        }
    }
}

// ==========================================
// MODAL FUNCTIONS
// ==========================================

function showWinnerModal(icon, title, message) {
    const modal = document.getElementById('winner-modal');
    const modalIcon = document.getElementById('modal-icon');
    const modalTitle = document.getElementById('modal-title');
    const modalMessage = document.getElementById('modal-message');
    
    modalIcon.textContent = icon;
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    
    modal.classList.add('show');
}

function closeModal() {
    const modal = document.getElementById('winner-modal');
    modal.classList.remove('show');
    startNewGame(false);
}

// ==========================================
// EVENT LISTENERS
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    // Add click handlers to cells
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, index) => {
        cell.addEventListener('click', () => makePlayerMove(index));
    });
    
    // Close modal on click outside
    const modal = document.getElementById('winner-modal');
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // Load scores from localStorage
    const savedScores = localStorage.getItem('tictactoe-scores');
    if (savedScores) {
        const scores = JSON.parse(savedScores);
        gameState.playerWins = scores.playerWins || 0;
        gameState.aiWins = scores.aiWins || 0;
        gameState.draws = scores.draws || 0;
        updateScore();
    }
    
    // Initial message
    showMessage('Click "New Game" to start playing!');
});

// Save scores to localStorage when they change
function updateScore() {
    document.getElementById('player-wins').textContent = gameState.playerWins;
    document.getElementById('ai-wins').textContent = gameState.aiWins;
    document.getElementById('draws').textContent = gameState.draws;
    
    localStorage.setItem('tictactoe-scores', JSON.stringify({
        playerWins: gameState.playerWins,
        aiWins: gameState.aiWins,
        draws: gameState.draws
    }));
}

// ==========================================
// KEYBOARD SHORTCUTS
// ==========================================

document.addEventListener('keydown', (e) => {
    // Press 'N' for new game
    if (e.key.toLowerCase() === 'n' && !gameState.gameOver) {
        startNewGame(false);
    }
    
    // Press numbers 1-9 to make moves
    const num = parseInt(e.key);
    if (num >= 1 && num <= 9) {
        makePlayerMove(num - 1);
    }
    
    // Press 'Escape' to close modal
    if (e.key === 'Escape') {
        const modal = document.getElementById('winner-modal');
        if (modal.classList.contains('show')) {
            closeModal();
        }
    }
});

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

// Add visual feedback for keyboard users
document.addEventListener('DOMContentLoaded', () => {
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell) => {
        cell.setAttribute('tabindex', '0');
        
        cell.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                cell.click();
            }
        });
    });
});
