import sys
sys.path.append('..')

from Game import Game
from QuoridorLogic import Board
import numpy as np

class QuoridorGame(Game):
    def __init__(self, board_size = 9, wall_count = 10):
        super().__init__()
        self.board_size = board_size  # Quoridor is played on a 9x9 grid
        self.wall_count = wall_count # Each player has 10 walls

    def getInitBoard(self):
        # Create the initial board setting
        # Representing pawns with 1 and 2, and empty spaces with 0
        # Walls will need a separate representation

        board = Board()
        return board

    def getBoardSize(self):
        return (self.board_size, self.board_size)

    def getActionSize(self):
        # There are 4 possible actions for pawns, and then there are 128 possible places for walls
        # (64 for horizontal and 64 for vertical), totaling 132.

        return 4 + 2 * ((self.board_size - 1) * (self.board_size - 1))

    def getNextState(self, board, player, action):
        # This is a simplified placeholder.
        # You need to implement how actions modify the board
        next_board = board.copy() # You'll need a more complex copy for actual wall state
        next_player = -player
        # Apply action to board to get next state
        return next_board, next_player

    def getValidMoves(self, board, player):
        # Return a list of valid moves
        # This will have to check the current board state and player positions/walls
        valid_moves = [0] * self.getActionSize()
        # Populate valid_moves with actual logic
        return valid_moves

    def getGameEnded(self, board, player):
        # Check if any player has reached the opposite side
        if some_condition_for_player1_win:
            return 1 if player == 1 else -1
        elif some_condition_for_player2_win:
            return -1 if player == -1 else 1
        else:
            return 0  # Game not ended

    def getCanonicalForm(self, board, player):
        # For Quoridor, the canonical form can simply be the board multiplied by the player,
        # so that the current player's pawns always appear as '1'
        canonical_board = [[x * player for x in row] for row in board]
        return canonical_board

    def getSymmetries(self, board, pi):
        # Assuming pi is a policy vector (list of probabilities for each action)
        # Quoridor has 4-fold rotational symmetry
        # Here you would implement code to rotate the board and pi to match
        # For now, we'll just return the unmodified board and pi in a list
        return [(board, pi)]

    def stringRepresentation(self, board):
        # Convert the board to a string
        return str(board)