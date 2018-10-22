import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetworkPongCNN(nn.Module):
    def __init__(self, input_shape, n_actions=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        # fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class A2CNetworkPongCNN(nn.Module):
    def __init__(self, input_shape, n_actions=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.h = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.fc_policy = nn.Sequential(
            nn.Linear(512, n_actions),
            nn.LogSoftmax(dim=-1)
        )
        self.fc_vf = nn.Sequential(
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # fx = x.float() / 256
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        h_out = self.h(conv_out)
        return self.fc_policy(h_out), self.fc_vf(h_out)


def main():
    net = A2CNetworkPongCNN((4,84,84))
    # from _PongEnv import Pong
    # env = Pong(gamma=0.9)
    # print(net(env.reset().unsqueeze(0)))
    print(net(torch.zeros((5,4,84,84))))

if __name__ == '__main__':
    main()