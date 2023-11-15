import sys
sys.path.append('../games')

import numpy as np

import tkinter as tk
from tkinter import ttk

# MCTS
from MCTS import MCTS

# othello
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as OthelloNet


class OthelloGUI():
    def __init__(self, master, mode, args):
        self.master = master

        self.game = OthelloGame(8)
        self.board = self.game.getInitBoard()

        hp = HumanOthelloPlayer(self.game).play
        neural_net = OthelloNet(self.game)
        neural_net.load_checkpoint('../pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')

        mcts = MCTS(self.game, neural_net, args)
        player1 = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
        player2 = hp

        self.player1 = player1
        self.player2 = player2

    def init_ui(self):
        # Initialize the UI components
        # Create a chessboard grid and place pieces

        # Add a label to display game status and rules
        rules_label = tk.Label(self.master, text="Welcome to Othello!\nClick on an empty cell to make a move.\n"
                                                 "Flank your opponent's pieces to capture them.\n"
                                                 "The game ends when the board is full or no legal moves are left.",
                               bg="#00796b", fg="#ffffff", font=("Helvetica", 10), justify="left")
        rules_label.pack()

        # Add a new canvas for valid move indicators
        canvas = tk.Canvas(self.master, width=400, height=400, bg="#00796b")
        canvas.pack()

        for row in range(8):
            for col in range(8):
                x0, y0 = col * 50, row * 50
                x1, y1 = x0 + 50, y0 + 50
                canvas.create_rectangle(x0, y0, x1, y1, fill="#00796b", outline="#004d40", width=2, tags="grid")

                if self.board[row][col] == "B":
                    canvas.create_image((x0 + x1) // 2, (y0 + y1) // 2, image=black_stone_image, tags="stones")
                elif self.board[row][col] == "W":
                    canvas.create_image((x0 + x1) // 2, (y0 + y1) // 2, image=white_stone_image, tags="stones")

                #if is_valid_move(row, col):
                #    x0, y0 = col * 50 + 20, row * 50 + 20
                #    x1, y1 = x0 + 10, y0 + 10
                #    canvas.create_oval(x0, y0, x1, y1, outline="yellow", width=2, tags="indicators")

        canvas.tag_raise("indicators")  # Ensure that indicators are drawn on top of stones



    def on_clicked(self, square):
        # Handle square click event
        # Determine the move and pass it to the game logic
        pass

    def update_board(self):
        # Update the GUI to reflect the current game state
        # self.board =
        pass

    def game_loop(self):
        # Main game loop
        # Check for game over conditions, switch turns, etc.



        pass