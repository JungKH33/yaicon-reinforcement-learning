import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class YachtDiceNNet(nn.Module):
    def __init__(self, game):
        super(YachtDiceNNet, self).__init__()
        self.state_size = game.getBoardSize()
        self.state_size = self.state_size[0] * self.state_size[1]
        self.action_size = game.getActionSize()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self.state_size, 500)
        #self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(500, 512)
        #self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = self.flatten(s)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
