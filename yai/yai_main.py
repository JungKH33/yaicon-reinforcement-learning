import tkinter as tk
from tkinter import ttk

from ArenaGUI import ArenaGUI

root = tk.Tk()
root.title("Game Selector")
root.geometry("400x300")  # Adjust the size as needed


# List of games
games = ['오셀로', '커넥트4', '틱택토', '3차원 틱택토', '전투오목', 'Dots and Boxes']
# List of difficulty levels
difficulties = ["매우 약함", "약함", "보통", "강함", "매우 강함", "신"]
# List of modes
modes = ["알파제로와 대국하기", "알파제로의 훈수받기"]

# Variable to store the selected game
selected_game = tk.StringVar()
# Variable to store the selected difficulty
selected_difficulty = tk.StringVar()
# Variable to store the selected mode
selected_mode = tk.StringVar()

# Create a drop-down menu for game selection
game_menu = ttk.OptionMenu(root, selected_game, games[0], *games)
game_menu.pack(pady=10)
# Create a drop-down menu for difficulty selection
difficulty_menu = ttk.OptionMenu(root, selected_difficulty, difficulties[0], *difficulties)
difficulty_menu.pack(pady=10)
# Create a drop-down menu for difficulty selection
mode_menu = ttk.OptionMenu(root, selected_mode, modes[0], *modes)
mode_menu.pack(pady=10)




def start_game():
    chosen_game = selected_game.get()
    chosen_difficulty = selected_difficulty.get()
    chosen_mode = selected_mode.get()
    print(f"Starting {chosen_game} with {chosen_difficulty} difficulty")

    game_window = tk.Toplevel(root)
    arena = ArenaGUI(master = game_window, game = chosen_game, mode = chosen_mode, difficulty = chosen_difficulty)
    game_window.mainloop()



start_button = ttk.Button(root, text="Start Game", command=start_game)
start_button.pack(pady=10)

root.mainloop()