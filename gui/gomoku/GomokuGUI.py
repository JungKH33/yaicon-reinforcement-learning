import sys
sys.path.append('../')
sys.path.append('../../games')
import os

import numpy as np

import tkinter as tk
import math
import time

from PIL import Image, ImageTk

# MCTS
from MCTS import MCTS

# gomoku
from games.gomoku.GomokuGame import GomokuGame
from games.gomoku.pytorch.NNet import NNetWrapper as GomokuNet

class GomokuGUI():
    def __init__(self, master, mode, args):
        self.master = master
        self.mode = mode

        self.board_size = 8
        self.num_in_row = 5
        self.game = GomokuGame(self.board_size, self.num_in_row)
        self.board = self.game.getInitBoard()

        player1 = Human()
        player2 = AlphaZero(self.game, args)

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

        # load player labels
        self.player_label = tk.Label(self.master, text=f"현재 플레이어: {'흑돌' if self.curPlayer == 1 else '백돌'}",
                                     bg="#00796b", fg="#ffffff", font=("Helvetica", 12))
        self.player_label.grid()

        # create canvas
        self.canvas = tk.Canvas(self.master, width=600, height=600)
        self.canvas.grid(row=0, column=0, padx=100, pady=100)

        self.init_ui()

    def load_and_resize_image(self, image_path, target_size):
        image = Image.open(image_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        return ImageTk.PhotoImage(image)

    def init_ui(self):
        self.draw_board()
        self.player_label.config(text=f"현재 플레이어: {'흑돌' if self.curPlayer == 1 else '백돌'}")

    def draw_board(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.canvas.create_rectangle(i * 40, j * 40, i * 40 + 40, j * 40 + 40, fill="white")
                self.canvas.tag_bind("rectangle", "<Button-1>", lambda event, row=i, col=j: self.place_stone(row, col))
    def place_stone(self, row, col):
        print('place')
        color = "black" if self.curPlayer == 1 else "white"
        self.canvas.create_oval(row * 40 + 5, col * 40 + 5, row * 40 + 35, col * 40 + 35, fill= color)
        self.toggle_player()

    def toggle_player(self):
        # self.curPlayer = "black" if self.curPlayer == "white" else "white"
        pass


class Human():
    def __init__(self):
        pass

    def __str__(self):
        return 'human'
class AlphaZero():
    def __init__(self, game, args):
        self.game = game

        neural_net = GomokuNet(self.game)
        neural_net.load_checkpoint('../../best_models/gomoku', 'gomoku_885_v3.pth.tar')
        self.mcts = MCTS(self.game, neural_net, args)

    def action(self, x):
        return np.argmax(self.mcts.getActionProb(x, temp=0))

    def __str__(self):
        return "ai"



if __name__ == "__main__":
    root = tk.Tk()
    args = {'numMCTSSims': 1500, 'cpuct': 2.0}
    gui = GomokuGUI(master = root, mode = None, args = args)
    root.mainloop()
