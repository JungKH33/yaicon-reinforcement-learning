class Board:
    def __init__(self, size=9):
        self.size = size  # Size for pawns' grid
        self.board_size = (2*size) - 1  # Size of the full board including walls
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.initialize_pawns()

    def initialize_pawns(self):
        # Initialize pawns at their starting positions
        mid = self.board_size // 2  # Middle index for pawn placement
        self.board[mid][0] = '1'  # Player 1 pawn represented by '1'
        self.board[mid][self.board_size - 1] = '2' # Player 2 pawn represented by '2'

    def add_wall(self, orientation, row, col):
        # Place a wall on the board, 'h' for horizontal, 'v' for vertical
        # We should convert 'row' and 'col' to the indices of the 17x17 board
        if orientation == 'h':
            wall_row = (row * 2) - 1
            wall_col_start = col * 2
            self.board[wall_row][wall_col_start] = self.board[wall_row][wall_col_start + 1] = self.board[wall_row][wall_col_start + 2] = 'H'
        elif orientation == 'v':
            wall_col = (col * 2) - 1
            wall_row_start = row * 2
            self.board[wall_row_start][wall_col] = self.board[wall_row_start + 1][wall_col] = self.board[wall_row_start + 2][wall_col] = 'V'

    def move_pawn(self, player, direction):
        # Move pawn for the given player in the specified direction if possible
        # direction is one of 'up', 'down', 'left', 'right'
        # Find the current position of the pawn
        pawn = '1' if player == 1 else '2'
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == pawn:
                    new_i, new_j = i, j
                    if direction == 'up':
                        new_i -= 2
                    elif direction == 'down':
                        new_i += 2
                    elif direction == 'left':
                        new_j -= 2
                    elif direction == 'right':
                        new_j += 2

                    # Check if the move is valid
                    if self.is_valid_move(i, j, new_i, new_j):
                        self.board[i][j], self.board[new_i][new_j] = 0, pawn
                        return True  # Move was successful
        return False  # Move was not made

    def is_valid_move(self, i, j, new_i, new_j):
        # Check if the new position is within bounds and not blocked by walls
        if not (0 <= new_i < self.board_size and 0 <= new_j < self.board_size):
            return False  # New position is out of bounds

        # Check if the move is horizontally or vertically adjacent (not diagonal)
        if abs(new_i - i) + abs(new_j - j) != 2:
            return False  # Invalid move distance

        # Check for walls
        if self.is_wall_between(i, j, new_i, new_j):
            return False  # Move is blocked by a wall

        return True  # All checks passed

    def is_wall_between(self, i, j, new_i, new_j):
        # Determine if a wall exists between the old and new positions
        # Wall checks depend on the direction of movement
        if new_i < i:  # Moving up
            return any(self.board[i - 1][j] == 'H' for k in range(-1, 2))
        elif new_i > i:  # Moving down
            return any(self.board[i + 1][j] == 'H' for k in range(-1, 2))
        elif new_j < j:  # Moving left
            return any(self.board[i][j - 1] == 'V' for k in range(-1, 2))
        elif new_j > j:  # Moving right
            return any(self.board[i][j + 1] == 'V' for k in range(-1, 2))
        return False  # No wall between positions

    def print_board(self):
        # Print the board state
        for row in self.board:
            print(" ".join(str(cell) for cell in row))

board = Board()
board.add_wall('v', 3,3)
board.move_pawn(1,'right')
board.move_pawn(1,'right')
board.move_pawn(1,'right')
board.print_board()