import sys
sys.path.append('../games')

import tkinter as tk
from tkinter import ttk

from gui.othello.OthelloGUI import OthelloGUI


class ArenaGUI():
    def __init__(self, master, game, mode, difficulty):
        self.master = master
        self.game = game
        self.mode = mode
        self.difficulty = difficulty
        args = performance_settings[self.difficulty]

        self.othello = OthelloGUI(self.master, self.mode, args)
        self.init_ui()

    def init_ui(self):
        # Initialize the UI components
        self.othello.init_ui()

    def on_clicked(self, square):
        # Handle square click event
        # Determine the move and pass it to the game logic
        self.othello.on_clicked()
        pass

    def update_board(self):
        # Update the GUI to reflect the current game state
        self.othello.update_board()
        pass

    def game_loop(self):
        # Main game loop
        # Check for game over conditions, switch turns, etc.
        self.othello.game_loop()
        pass


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

performance_settings = {
    '매우 약함': dotdict({'numMCTSSims': 50, 'cpuct': 2.0}),
    '약함': dotdict({'numMCTSSims': 100, 'cpuct': 2.0}),
    '중간': dotdict({'numMCTSSims': 300, 'cpuct': 2.0}),
    '강함': dotdict({'numMCTSSims': 700, 'cpuct': 2.0}),
    '매우 강함': dotdict({'numMCTSSims': 1000, 'cpuct': 2.0}),
    '신': dotdict({'numMCTSSims': 1500, 'cpuct': 2.0})
}