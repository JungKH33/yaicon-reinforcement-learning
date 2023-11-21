import tkinter
import math
import time
import numpy as np

# MCTS
from MCTS import MCTS

# gomoku
from games.gomoku.GomokuGame import GomokuGame
from games.gomoku.pytorch.NNet import NNetWrapper as GomokuNet

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
class AlphaZero():
    def __init__(self, game):
        self.game = game

        neural_net = GomokuNet(self.game)
        neural_net.load_checkpoint('../../best_models/gomoku/', 'gomoku_885_v5.pth.tar')
        self.mcts = MCTS(self.game, neural_net, args = dotdict({'numMCTSSims': 1500, 'cpuct': 2.0}) )
    def get_move(self, x):
        action = np.argmax(self.mcts.getActionProb(x, temp=0))
        move = (action % self.game.n, action // self.game.n)
        return move
    def get_all(self, x):
            policy, action_value_f, value_f = self.mcts.getallValues(x)
            return policy, action_value_f, value_f
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pixel_x = 30 + 30 * self.x
        self.pixel_y = 30 + 30 * self.y

class BoardCanvas(tkinter.Canvas):
    def __init__(self, master=None, height=0, width=0, board_size=15):
        self.board_size = board_size
        tkinter.Canvas.__init__(self, master, height=height, width=width)
        self.init_board_points()
        self.init_board_canvas()

    def init_board_points(self):
        self.board_points = [[None for i in range(self.board_size)] for j in range(self.board_size)]
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board_points[i][j] = Point(i, j)

    def init_board_canvas(self):
        for i in range(self.board_size):
            p1 = self.board_points[i][0]
            p2 = self.board_points[i][self.board_size - 1]
            self.create_line(p1.pixel_x, p1.pixel_y, p2.pixel_x, p2.pixel_y)

        for j in range(self.board_size):
            p1 = self.board_points[0][j]
            p2 = self.board_points[self.board_size - 1][j]
            self.create_line(p1.pixel_x, p1.pixel_y, p2.pixel_x, p2.pixel_y)

        for i in range(self.board_size):
            for j in range(self.board_size):
                r = 1
                p = self.board_points[i][j]
                self.create_oval(p.pixel_x - r, p.pixel_y - r, p.pixel_x + r, p.pixel_y + r)

    def place_move(self, move, color):
        """Draw circle on the board"""
        p = self.board_points[move[0]][move[1]]
        self.create_oval(p.pixel_x - 10, p.pixel_y - 10, p.pixel_x + 10, p.pixel_y + 10, fill=color)

    def find_move_coordinates(self, event):
        """Find line crossing closest to the click position"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                p = self.board_points[i][j]
                square_distance = math.pow((event.x - p.pixel_x), 2) + math.pow((event.y - p.pixel_y), 2)
                if (square_distance <= 200):
                    return (i, j)

    def print_message(self, text):
        """Display text mesaage below board"""
        self.delete("text_tag")
        self.create_text(240, 550, text=text, tag="text_tag")

class BoardFrame(tkinter.Frame):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.board_label_frame = tkinter.LabelFrame(self, text="Gomoku", padx=5, pady=5)
        self.board_label_frame.pack()

class BoardGame(object):
    """Board game to paly computer vs human.

    Please note that board x coordinate is horisontal from left to right,
    board y coordinate is vertical from top to bottom."""

    def __init__(self):
        self.board_size = 8
        self.num_in_row = 5
        self.game = GomokuGame(self.board_size, self.num_in_row)

        # initialize players and board
        self.player = AlphaZero(self.game)
        self.curPlayer = 1
        self.board = self.game.getInitBoard()

        # history
        self.board_history = []
        self.policy_history = []
        self.Vs_history = []
        self.Qsa_history = []


        window = tkinter.Tk()
        self.board_frame = BoardFrame(window)
        self.board_canvas = BoardCanvas(self.board_frame.board_label_frame, height=600, width=480,
                                        board_size= self.board_size)

        # bind left mouse button click event
        self.board_canvas.bind('<Button-1>', self.click_event)

        self.board_frame.pack()
        self.board_canvas.pack()

        # do first move for computer if she is black
        #if False:
        #    move = self.player.get_move(self.board)
        #    self.process_move(move)

        window.mainloop()

    def process_move(self, move):
        # if move is None:
        #    self.board_canvas.print_message("Draw")
        #    self.board_canvas.unbind('<Button-1>')
        #    return None

        # check and record move, if correct

        x, y = move
        action = self.board_size * y + x
        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.board, self.curPlayer), 1)

        if valids[action] == 0:
            self.board_canvas.print_message("Move is invalid!")
            return False

        policyf, actionf, valuef = self.player.get_all(self.game.getCanonicalForm(self.board, self.curPlayer))
        self.board_history.append(self.board.copy())
        self.policy_history.append(policyf)
        self.Vs_history.append(valuef)
        self.Qsa_history.append(actionf)
        print(valuef)

        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

        player_color = 'black'
        if self.curPlayer == 1:
            player_color = 'white'

        # display move on board
        self.board_canvas.place_move(move, player_color)
        # self.board_canvas.print_message("Player " + player_color + " move " + str(move))
        # time.sleep(0.1)
        self.board_canvas.print_message("")  # clear error message if any

        # check if this is the winning move (or draw)
        player_wins = self.game.getGameEnded(self.board, self.curPlayer)

        if player_wins == 1 or player_wins == - 1:
            self.finish_game("Player " + player_color + " wins")
            return False

        if player_wins != 0:  # draw
            self.finish_game("Draw")
            return False

        return True

    def finish_game(self, message):
        self.board_canvas.print_message(message)
        self.board_canvas.unbind('<Button-1>')

    def click_event(self, event):
        """Wait for human to make a move, then palce computer move"""

        move = self.board_canvas.find_move_coordinates(event)

        if not self.process_move(move):
            return  # wait for another click

        move = self.player.get_move(self.board)
        time.sleep(0.1)
        self.process_move(move)  # assume computer move is always valid

if __name__ == '__main__':

    board = BoardGame()