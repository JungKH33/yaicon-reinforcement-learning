import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, filters, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        out = F.relu(x + y)
        return out


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class PolicyHead(nn.Module):
    def __init__(self, action_size):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.fc = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.softmax(self.fc(x), dim=1)
        return x


class Connect4NNet(nn.Module):
    def __init__(self, game, args):
        super(Connect4NNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.conv1 = nn.Conv2d(1, args.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(args.num_channels) for _ in range(args.num_residual_layers)])
        self.value_head = ValueHead()
        self.policy_head = PolicyHead(self.action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)  # Reshape to match PyTorch's NCHW format
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        pi = self.policy_head(x)
        v = self.value_head(x)
        return pi, v

    def loss_pi(self, targets, outputs):
        return F.cross_entropy(outputs, targets)

    def loss_v(self, targets, outputs):
        return F.mse_loss(outputs.view(-1), targets)