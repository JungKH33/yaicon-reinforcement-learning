from __future__ import print_function
import sys
sys.path.append('..')

import random, os
import numpy as np
from Game import Game

class YachtDiceGame(Game):
    def __init__(self):
        self.boardname = ['Aces\t', 'Deuces', 'Threes', 'Fours', 'Fives', 'Sixes', 'Choice'
                          , '4 of a Kind', 'Full House', 'S. Straight', 'L. Straight', 'Yacht']
        self.idxmapping = {0:[0, 0, 0, 0, 0], 1:[1, 0, 0, 0, 0], 2:[0, 1, 0, 0, 0], 3:[0, 0, 1, 0, 0], 4:[0, 0, 0, 1, 0], 5:[0, 0, 0, 0, 1], 6:[1, 1, 0, 0, 0], 7:[1, 0, 1, 0, 0],
                           8:[1, 0, 0, 1, 0], 9:[1, 0, 0, 0, 1], 10:[0, 1, 1, 0, 0], 11:[0, 1, 0, 1, 0], 12:[0, 1, 0, 0, 1], 13:[0, 0, 1, 1, 0], 14:[0, 0, 1, 0, 1], 15:[0, 0, 0, 1, 1],
                           16:[1, 1, 1, 0, 0], 17:[1, 1, 0, 1, 0], 18:[1, 1, 0, 0, 1], 19:[1, 0, 1, 1, 0], 20:[1, 0, 1, 0, 1], 21:[1, 0, 0, 1, 1], 22:[0, 1, 1, 1, 0], 23:[0, 1, 1, 0, 1],
                           24:[0, 1, 0, 1, 1], 25:[0, 0, 1, 1, 1], 26:[1, 1, 1, 1, 0], 27:[1, 1, 1, 0, 1], 28:[1, 1, 0, 1, 1], 29:[1, 0, 1, 1, 1], 30:[0, 1, 1, 1, 1], 31:[1, 1, 1, 1, 1]}
        self.action_size = 32
        self.state_size = 23
        self.panelty = -1

    def getInitBoard(self):
        return self.reset()
    
    def getBoardSize(self):
        return (2, self.state_size)
    
    def getActionSize(self):
        return self.action_size
    
    def updateState(self, board):
        state1 = board[0]
        state2 = board[1]
        self.scoreboard1 = state1[:12]
        self.hand1 = state1[12:17]
        self.hand_mask1 = state1[17:22]
        self.rollcount1 = state1[-1]

        self.scoreboard2 = state2[:12]
        self.hand2 = state2[12:17]
        self.hand_mask2 = state2[17:22]
        self.rollcount2 = state2[-1]

    def getNextState(self, board, player, action):
        self.updateState(board)
        if player == 1:
            valid = np.array(self.getValidMoves(board, 1))
            # action = np.argmax(action)
            if action in np.where(valid == 1)[0]:
                if self.rollcount1 > 0:
                    save_idx = self.idxmapping[action]
                    for s in range(5):
                        if save_idx[s] == 1:
                            self.hand_mask1[s] = 1
                    self.roll(1)
                else:
                    place_idx = action
                    self.scoreboard1[place_idx] = self.calculate_score(place_idx)
                    self.rollcount1 = 3
                    self.hand_mask1 = [-1] * 5
                    self.roll(1)
            else:
                return self._get_obs(), 1
            return self._get_obs(), -1
        else:
            valid = np.array(self.getValidMoves(board, -1))
            # action = np.argmax(action)

            if action in np.where(valid == 1)[0]:
                if self.rollcount2 > 0:
                    save_idx = self.idxmapping[action]
                    for s in range(5):
                        if save_idx[s] == 1:
                            self.hand_mask2[s] = 1
                    self.roll(-1)
                else:
                    place_idx = action
                    self.scoreboard2[place_idx] = self.calculate_score(place_idx)
                    self.rollcount2 = 3
                    self.hand_mask2 = [-1] * 5
                    self.roll(-1)
            else:
                return self._get_obs(), 1
            return self._get_obs(), -1
        
        
    def getValidMoves(self, board, player):
        self.updateState(board)
        if player == 1:
            if self.rollcount1 == 0:
                idxs = [1 if s == -1 else 0 for s in self.scoreboard1]
                return idxs + [0] * 20
            else:
                idxs = []
                for i, idx in enumerate(self.idxmapping.values()):
                    passed = True
                    for j in range(5):
                        if self.hand_mask1[j] == 1 and idx[j] == 1:
                            passed = False
                            break
                    if passed:
                        idxs.append(i)
                act = self.get_dummy_action()
                for i in range(32):
                    if i in idxs:
                        act[i] = 1
                return act
        else:
            if self.rollcount2 == 0:
                idxs = [1 if s == -1 else 0 for s in self.scoreboard2]
                return idxs + [0] * 20
            else:
                idxs = []
                for i, idx in enumerate(self.idxmapping.values()):
                    passed = True
                    for j in range(5):
                        if self.hand_mask2[j] == 1 and idx[j] == 1:
                            passed = False
                            break
                    if passed:
                        idxs.append(i)
                act = self.get_dummy_action()
                for i in range(32):
                    if i in idxs:
                        act[i] = 1
                return act


    def getGameEnded(self, board, player):
        self.updateState(board)
        flag1 = True
        flag2 = True

        for i in self.scoreboard1:
            if i == -1:
                flag1 = False
                break
        for i in self.scoreboard2:
            if i == -1:
                flag2 = False
                break
        if flag1 and flag2:
            s1 = np.sum(self.scoreboard1)
            s2 = np.sum(self.scoreboard2)
            return 1 if s1 > s2 else -1
        return 0

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        else:
            temp = np.array(board)
            temp[0] = board[1]
            temp[1] = board[0]
            return temp

    def getSymmetries(self, board, pi):
        return [board, pi]

    def stringRepresentation(self, board):
        self.updateState(board)
        return self.render()

    def get_dummy_action(self):
        return [0] * self.action_size
    
    def get_subtotal(self, player):
        stack = 0
        if player == 1:
            for j in self.scoreboard1[:6]:
                if j != -1:
                    stack += j
        else:
            for j in self.scoreboard2[:6]:
                if j != -1:
                    stack += j
        return stack
    
    def get_total(self, player):
        stack = 0
        if player == 1:
            for j in self.scoreboard1:
                if j != -1:
                    stack += j
            if np.sum(self.scoreboard1[:6]) >= 63:
                stack += 35
        else:
            for j in self.scoreboard2:
                if j != -1:
                    stack += j
            if np.sum(self.scoreboard2[:6]) >= 63:
                stack += 35 
        return stack       

    def reset(self):
        self.scoreboard1 = [-1] * 12
        self.hand1 = [-1] * 5
        self.hand_mask1 = [-1] * 5
        self.rollcount1 = 3

        self.scoreboard2 = [-1] * 12
        self.hand2 = [-1] * 5
        self.hand_mask2 = [-1] * 5
        self.rollcount2 = 3

        self.roll(1)
        self.roll(-1)

        return self._get_obs()

    def render(self, cls=True):
        if cls:
            os.system('cls')
        self._show_board()

    def _get_obs(self):
        state1 = self.scoreboard1+self.hand1+self.hand_mask1+[self.rollcount1]
        state2 = self.scoreboard2+self.hand2+self.hand_mask2+[self.rollcount2]
        return np.array([state1, state2])

    def roll(self, player):
        if player == 1:
            if self.rollcount1 > 0:
                idx = []
                for i in range(5):
                    if self.hand_mask1[i] != 1:
                        idx.append(i)
                for i in idx:
                    self.hand1[i] = random.randint(1, 6)
                self.rollcount1 -= 1
                if self.rollcount1 == 0:
                    self.hand_mask1 = [1] * 5
        else:
            if self.rollcount2 > 0:
                idx = []
                for i in range(5):
                    if self.hand_mask2[i] != 1:
                        idx.append(i)
                for i in idx:
                    self.hand2[i] = random.randint(1, 6)
                self.rollcount2 -= 1
                if self.rollcount2 == 0:
                    self.hand_mask2 = [1] * 5

    def _show_board(self):
        print('Yacht Dice by jellyho')
        print("====================================="*2)
        for i in range(6):
            print(f"{i+1}. {self.boardname[i]}\t\t{'-' if self.scoreboard1[i] == -1 else self.scoreboard1[i]}\t\t", end="")
            print(f"{i+1}. {self.boardname[i]}\t\t{'-' if self.scoreboard2[i] == -1 else self.scoreboard2[i]}")
        print(f"Subtotal : {self.get_subtotal(1)}/63\t{'+35' if self.get_subtotal(1) >= 63 else ''}\t\t\t", end="")
        print(f"Subtotal : {self.get_subtotal(-1)}/63\t{'+35' if self.get_subtotal(-1) >= 63 else ''}")
        print("-------------------------------------"*2)
        for i in range(6, 12):
            print(f"{i+1}. {self.boardname[i]}\t\t{'-' if self.scoreboard1[i] == -1 else self.scoreboard1[i]}\t\t", end="")
            print(f"{i+1}. {self.boardname[i]}\t\t{'-' if self.scoreboard2[i] == -1 else self.scoreboard2[i]}")
        print("-------------------------------------"*2)
        mask1 = [str(i+1) if self.hand_mask1[i] == -1 else '-' for i in range(5)]
        hand_mask1 = ''
        for m in mask1:
            hand_mask1 += m + '  '
        mask2 = [str(i+1) if self.hand_mask2[i] == -1 else '-' for i in range(5)]
        hand_mask2 = ''
        for m in mask2:
            hand_mask2 += m + '  '
        print(f'Total : {self.get_total(1)}\t\t\t\t', end="")
        print(f'Total : {self.get_total(-1)}')
        print(f"`````````{hand_mask1[:-2]}```````````````", end="")
        print(f"`````````{hand_mask2[:-2]}```````````````")
        print('Dice : ', self.hand1, ' Reroll:', self.rollcount1, '  ', end="")
        print('Dice : ', self.hand2, ' Reroll:', self.rollcount2, '  ')
        print("====================================="*2)
        print()

    def calculate_score(self, i):
        dices = np.array(self.hand)
        if i in list(range(6)): #Aces ~ Sixes
            return np.sum(dices[np.where(dices==i+1)])
        elif i == 6: # Choice
            return np.sum(dices)
        elif i == 7: # 4 of a Kind
            for i in range(1, 7):
                if len(np.where(dices==i)[0]) >= 4:
                    return np.sum(dices)
            return 0
        elif i == 8: # Full House
            self.hand.sort()
            lst = self.hand
            counts = {}
            for num in lst:
                if num in counts:
                    counts[num] += 1
                else:
                    counts[num] = 1
            if len(counts) == 2 and 3 in counts.values():
                return np.sum(dices)
            return 0
        elif i == 9: # S Staright
            self.hand.sort()
            sorted = self.hand
            count = [0] * 6
            for s in sorted:
                count[s-1] += 1
            for i in range(2):
                passed = True
                for j in range(3):
                    if count[i + j] > 0 and count[i + j + 1] > 0:
                        continue
                    else:
                        passed = False
                        break
                if passed:
                    return 15
            return 0
        elif i == 10: # L Straight
            self.hand.sort()
            sorted = self.hand
            for h in range(4):
                if sorted[h] + 1 != sorted[h + 1]:
                    return 0
            return 30
        elif i == 11: # Yacht
            for h in range(1, 7):
                if len(np.where(dices==h)[0]) == 5:
                    return 50
            return 0