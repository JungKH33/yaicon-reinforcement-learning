import sys
sys.path.append('..')

import Arena
import Arena_Advice
from MCTS import MCTS

# othello
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as OthelloNet

# connect 4
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.keras.NNet import NNetWrapper as Connect4Net

# tictactoe
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
from tictactoe.keras.NNet import NNetWrapper as TicTacToeNet

import numpy as np
from utils import *


print("\n알파제로에 오신 것을 환영합니다! 게임 목록은 다음과 같습니다")

## Game Choice
games = ['오셀로', '커넥트4', '틱택토', '3차원 틱택토', '산토리니']
for i in range(len(games)):
    print(f"{i}. {games[i]}")

game_choice = input("게임을 선택하려면 숫자를 선택하세요: ")
valid_inputs = [str(i) for i in range(len(games))]
while game_choice not in valid_inputs:
    game_choice = input("Invalid input, enter a number listed above: ")

## Game mode choice
mode_options = [
                "알파제로와 대국하기",
                "알파제로의 훈수받기",
                "Exit",
            ]

print()
for i in range(len(mode_options)):
    print(f"{i}. {mode_options[i]}")

mode_choice = input("게임 모드를 고르세요: ")
valid_inputs = [str(i) for i in range(len(mode_options))]
while mode_choice not in valid_inputs:
    mode_choice = input("Invalid input, enter a number listed above: ")

performance_options = [
                "매우 약함",
                "약함",
                "보통",
                "강함",
                "매우 강함",
                "신"
            ]

print()
for i in range(len(performance_options)):
    print(f"{i}. {performance_options[i]}")

performance_choice = input("알파제로의 성능을 고르세요: ")
valid_inputs = [str(i) for i in range(len(performance_options))]
while performance_choice not in valid_inputs:
    performance_choice = input("Invalid input, enter a number listed above: ")




## switch statements
if game_choice == '0':
    game = OthelloGame(8)
    display = OthelloGame.display

    gp = GreedyOthelloPlayer(game).play
    hp = HumanOthelloPlayer(game).play
    neural_net = OthelloNet(game)
    neural_net.load_checkpoint('../pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')

elif game_choice == '1':
    game = Connect4Game()
    display = Connect4Game.display

    gp = OneStepLookaheadConnect4Player(game).play
    hp = HumanOthelloPlayer(game).play
    neural_net = Connect4Net(game)


elif game_choice == '2':
    game = TicTacToeGame()
    display = TicTacToeGame.display

    gp = OneStepLookaheadConnect4Player(game).play
    hp = HumanTicTacToePlayer(game).play
    neural_net = TicTacToeNet(game)
    neural_net.load_checkpoint('../pretrained_models/tictactoe/keras/', 'best-25eps-25sim-10epch.pth.tar')


if performance_choice == '0':
    args = dotdict({'numMCTSSims': 10, 'cpuct': 1.0})

elif performance_choice == '1':
    args = dotdict({'numMCTSSims': 30, 'cpuct': 1.0})

elif performance_choice == '2':
    args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})

elif performance_choice == '3':
    args = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})

elif performance_choice == '4':
    args = dotdict({'numMCTSSims': 500, 'cpuct': 1.0})

elif performance_choice == '5':
    args = dotdict({'numMCTSSims': 1000, 'cpuct': 1.0})

##


if mode_choice == '0':
    mcts = MCTS(game, neural_net, args)
    player1 = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    player2 = hp

elif mode_choice == '1':
    mcts = MCTS(game, neural_net, args)
    player1 = hp
    player2 = hp
##



# play game
#arena = Arena.Arena(alphazero, player2, game, display= display)
arena = Arena_Advice.Arena(player1, player2, game, mcts, display= display)
print(arena.playGames(2, verbose=True))