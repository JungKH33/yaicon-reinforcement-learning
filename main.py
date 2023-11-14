import logging
import sys

sys.path.append('./games')
import coloredlogs

from Coach import Coach

# othello
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as OthelloNet

# connect 4
from connect4.Connect4Game import Connect4Game
from connect4.pytorch.NNet import NNetWrapper as Connect4Net

# gobang
from gobang import GobangGame
from gobang.pytorch.NNet import NNetWrapper as GobangNet

# quoridor
#from quoridor.QuoridorGame import QuoridorGame
#from quoridor.pytorch.NNet import NNetWrapper as QuoridorNet

# gomoku

# yatch
from yacht.YachtDiceGame import YachtDiceGame
from yacht.pytorch.NNet import NNetWrapper as YachtDiceNet

# tafl

# dots and boxes
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.pytorch.NNet import NNetWrapper as DotboxNet

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./pretrained_models/gomoku','best_policy_8_8_5.model'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', GobangGame.__name__)
    #g = OthelloGame(8)
    #g = Connect4Game()
    #g = GobangGame(8)
    #g = QuoridorGame(5)
    #g = YachtDiceGame()
    #g = TaflGame()
    g = DotsAndBoxesGame(3)


    log.info('Loading %s...', GobangNet.__name__)
    #nnet = OthelloNet(g)
    #nnet = Connect4Net(g)
    #nnet = GobangNet(g)
    #nnet = QuoridorGame(g)
    #nnet = GomokuNet(g)
    #nnet = YachtNet(g)
    nnet = DotboxNet(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        #c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
