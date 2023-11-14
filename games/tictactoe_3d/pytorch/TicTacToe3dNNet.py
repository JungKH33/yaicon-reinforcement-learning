import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        super(TicTacToeNNet, self).__init__()
        self.board_z, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv1 = nn.Conv3d(1, args.num_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv3d(args.num_channels, args.num_channels, kernel_size=3, padding='same')
        self.conv3 = nn.Conv3d(args.num_channels, args.num_channels, kernel_size=3, padding='same')
        self.conv4 = nn.Conv3d(args.num_channels, args.num_channels, kernel_size=3, padding='valid')

        self.bn1 = nn.BatchNorm3d(args.num_channels)
        self.bn2 = nn.BatchNorm3d(args.num_channels)
        self.bn3 = nn.BatchNorm3d(args.num_channels)
        self.bn4 = nn.BatchNorm3d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 2) * (self.board_y - 2) * (self.board_z - 2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.pi_head = nn.Linear(512, self.action_size)
        self.v_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.board_z, self.board_x,
                   self.board_y)  # reshape to (batch_size, channels, depth, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # flatten the tensor

        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.dropout(x, p=self.args.dropout, training=self.training)  # apply dropout

        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = F.dropout(x, p=self.args.dropout, training=self.training)  # apply dropout

        pi = F.softmax(self.pi_head(x), dim=1)  # get probabilities for actions
        v = torch.tanh(self.v_head(x))  # get value estimation

        return pi, v