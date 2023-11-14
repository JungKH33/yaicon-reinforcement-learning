import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
    
class HumanYachtDicePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        action = self.game.get_dummy_action()
        valid = np.array(self.getValidMoves(board, 1))
        done = False
        while not done:
            if board[0][-1] != 0:
                idx = input('input save indexes (1 ~ 5) : ').split(' ')
                if idx != '':
                    mask = [0] * 5
                    for i in range(5):
                        if str(i+1) in idx:
                            mask[i] = 1
                    actidx = self.game.getActionsfromMask(mask)
                    if actidx in valid:
                        action[actidx] = 1
                        return actidx
                    else:
                        print('Invalid Input')
                        continue
            place = int(input('input index of space that you want to place : '))
            if place in valid:
                action[place-1] = 1
                return place-1
            else:
                print('Invalid Input')
                continue