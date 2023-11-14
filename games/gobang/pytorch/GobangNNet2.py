import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """ 卷积块 """

    def __init__(self, in_channels: int, out_channel: int, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return F.relu(self.batch_norm(self.conv(x)))


class ResidueBlock(nn.Module):
    """ 残差块 """

    def __init__(self, in_channels=128, out_channels=128):
        """
        Parameters
        ----------
        in_channels: int
            输入图像通道数

        out_channels: int
            输出图像通道数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        return F.relu(out + x)


class PolicyHead(nn.Module):
    """ 策略头 """

    def __init__(self, in_channels=128, board_len=9):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        board_len: int
            棋盘大小
        """
        super().__init__()
        self.board_len = board_len
        self.in_channels = in_channels
        self.conv = ConvBlock(in_channels, 2, 1)
        self.new_fc = nn.Linear(2*board_len**2, board_len**2 + 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.new_fc(x.flatten(1))
        return F.log_softmax(x, dim=1)


class ValueHead(nn.Module):
    """ 价值头 """

    def __init__(self, in_channels=128, board_len=9):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        board_len: int
            棋盘大小
        """
        super().__init__()
        self.in_channels = in_channels
        self.board_len = board_len
        self.conv = ConvBlock(in_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(board_len**2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x


class GobangNNet2(nn.Module):
    """ 策略价值网络 """

    def __init__(self, game, args):
        """
        Parameters
        ----------
        board_len: int
            棋盘大小

        n_feature_planes: int
            输入图像通道数，对应特征
        """
        super().__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.board_len = self.board_x

        self.n_feature_planes = 6

        self.new_conv = ConvBlock(1, 128, 3, padding=1)
        self.residues = nn.Sequential(
            *[ResidueBlock(128, 128) for i in range(4)])
        self.policy_head = PolicyHead(128, self.board_len)
        self.value_head = ValueHead(128, self.board_len)

    def forward(self, x):
        """ 前馈，输出 `p_hat` 和 `V`

        Parameters
        ----------
        x: Tensor of shape (N, C, H, W)
            棋局的状态特征平面张量

        Returns
        -------
        p_hat: Tensor of shape (N, board_len^2)
            对数先验概率向量

        value: Tensor of shape (N, 1)
            当前局面的估值
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.new_conv(x)
        x = self.residues(x)
        p_hat = self.policy_head(x)
        value = self.value_head(x)
        return p_hat, value
