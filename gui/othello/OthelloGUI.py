import sys
sys.path.append('../')
import os

import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from PIL import Image, ImageTk

# MCTS
from MCTS import MCTS

# othello
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as OthelloNet


class OthelloGUI():
    def __init__(self, master, mode, args):
        self.master = master

        self.board_size = 8
        self.game = OthelloGame(self.board_size)
        self.board = self.game.getInitBoard()

        player1 = AlphaZero(self.game, args)
        player2 = Human()

        # load players
        self.player1 = player1
        self.player2 = player2
        self.players = [self.player2, None, self.player1]
        self.curPlayer = 1
        self.turn = 0

        # load images
        target_size = (50, 50)
        current_directory = os.path.dirname(__file__)

        self.blackstone = self.load_and_resize_image(os.path.join(current_directory, 'blackstone.png'), target_size)
        self.whitestone = self.load_and_resize_image(os.path.join(current_directory, 'whitestone.png'), target_size)
        self.flippedstones = []

        # load player labels
        self.player_label = tk.Label(self.master, text=f"현재 플레이어: {'흑돌' if self.curPlayer == 1 else '백돌'}",
                                bg="#00796b", fg="#ffffff", font=("Helvetica", 12))
        self.player_label.pack()


        # Add a new canvas for valid move indicators
        self.canvas = tk.Canvas(self.master, width=400, height=400, bg="#00796b")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_clicked)

    def init_ui(self):
        # Initialize the UI components
        # Create a chessboard grid and place pieces

        # Add a label to display game status and rules
        rules_label = tk.Label(self.master, text="Welcome to Othello!\nClick on an empty cell to make a move.\n"
                                                 "Flank your opponent's pieces to capture them.\n"
                                                 "The game ends when the board is full or no legal moves are left.",
                               bg="#00796b", fg="#ffffff", font=("Helvetica", 10), justify="left")
        rules_label.pack()

        # draw initial board
        self.draw_board()
        self.player_label.config(text=f"현재 플레이어: {'흑돌' if self.curPlayer == 1 else '백돌'}")

        # Add a start button
        self.start_button = tk.Button(self.master, text="Start Game", command= self.start_game)
        self.start_button.pack(pady=20)  # Add some vertical padding

    def on_clicked(self, event):
        if str(self.players[self.curPlayer + 1]) == 'ai':
            pass

        else:
            # Handle square click event
            col = event.x // 50
            row = event.y // 50

            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                if self.board[row][col] == 0 :
                    self.game_loop(row * self.board_size + col)

            self.canvas.bind("<Button-1>", self.on_clicked)

    def start_game(self):
        print("Game Started!")

        self.start_button.pack_forget()
        if str(self.players[self.curPlayer + 1]) == 'ai':
            self.game_loop()

    def update_board(self):
        # Update the GUI to reflect the current game state
        self.canvas.delete("all")
        self.draw_board()
    def game_loop(self, action = None):
        # Main game loop
        # Check for game over conditions, switch turns, etc.
        print("Turn ", str(self.turn), "Player ", str(self.curPlayer))

        if action is None:
            action = self.players[self.curPlayer + 1].action(self.game.getCanonicalForm(self.board, self.curPlayer))

        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.board, self.curPlayer), 1)

        if valids[action] == 0:
            messagebox.showinfo("Invalid Move", "This move is not valid. Try again.")
            # assert valids[action] > 0
            return False

        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)
        self.update_board()

        if self.game.getGameEnded(self.board, self.curPlayer) != 0:
            print("Game over: Turn ", str(self.turn), "Result ", str(self.game.getGameEnded(self.board, 1)))
            self.update_board()
            self.master.quit()
            return True

        self.turn += 1

        if str(self.players[self.curPlayer + 1]) == 'ai':
            self.game_loop()
    def load_and_resize_image(self, image_path, target_size):
        image = Image.open(image_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        return ImageTk.PhotoImage(image)

    def draw_board(self):
        valid_moves = self.game.getValidMoves(self.board, self.curPlayer)
        for row in range(self.board_size):
            for col in range(self.board_size):
                x0, y0 = col * 50, row * 50
                x1, y1 = x0 + 50, y0 + 50
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#00796b", outline="#004d40", width=2, tags="grid")

                if self.board[row][col] == 1 :
                    self.canvas.create_image((x0 + x1) // 2, (y0 + y1) // 2, image= self.blackstone, tags="stones")
                elif self.board[row][col] == -1 :
                    self.canvas.create_image((x0 + x1) // 2, (y0 + y1) // 2, image= self.whitestone, tags="stones")

                if valid_moves[row * self.board_size + col]:
                    x0, y0 = col * 50 + 20, row * 50 + 20
                    x1, y1 = x0 + 10, y0 + 10
                    self.canvas.create_oval(x0, y0, x1, y1, outline="yellow", width=2, tags="indicators")

        self.canvas.tag_raise("indicators")  # Ensure that indicators are drawn on top of stones


class Human():
    def __init__(self):
        pass

    def __str__(self):
        return 'human'

class AlphaZero():
    def __init__(self, game, args):
        self.game = game

        neural_net = OthelloNet(self.game)
        neural_net.load_checkpoint('../pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
        self.mcts = MCTS(self.game, neural_net, args)

    def action(self, x):
        return np.argmax(self.mcts.getActionProb(x, temp=0))

    def __str__(self):
        return "ai"
