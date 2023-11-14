import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class DotsAndBoxesNNet(nn.Module):
    def __init__(self, game, args):
        super(DotsAndBoxesNNet, self).__init__()
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.dropout_rate = args.dropout
        self.fc1 = nn.Linear(self.board_x * self.board_y, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.pi_head = nn.Linear(512, self.action_size)
        self.v_head = nn.Linear(512, 1)

    def forward(self, x):
        # Assuming x is of shape (batch_size, board_x, board_y)
        # Flatten the input
        x = x.view(x.size(0), -1)

        # First fully connected layer
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.fc3(x)), p=self.dropout_rate, training=self.training)
        x = F.dropout(F.relu(self.fc4(x)), p=self.dropout_rate, training=self.training)

        # Policy head
        pi = F.softmax(self.pi_head(x), dim=1)
        # Value head
        v = torch.tanh(self.v_head(x))

        return pi, v