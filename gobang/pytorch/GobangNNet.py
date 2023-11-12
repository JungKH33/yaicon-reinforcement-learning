import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class GobangNNet(nn.Module):
    def __init__(self, game, args):
        super(GobangNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.conv1 = nn.Conv2d(1, args.num_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, padding='valid')
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, padding='valid')

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.pi = nn.Linear(512, self.action_size)
        self.v = nn.Linear(512, 1)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, s):
        # s: batch_size x 1 x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # reshape the board

        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))

        s = s.view(s.size(0), -1)  # flatten the tensor

        s = self.dropout(F.relu(self.fc_bn1(self.fc1(s))))
        s = self.dropout(F.relu(self.fc_bn2(self.fc2(s))))

        pi = F.softmax(self.pi(s), dim=1)
        v = torch.tanh(self.v(s))

        return pi, v