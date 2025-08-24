import torch
import torch.nn as nn
import torch.nn.functional as F



# Residual trunk 
class ResidualBlock(nn.Module):
    def __init__(self, channels, bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not bn)
        self.bn1   = nn.BatchNorm2d(channels) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not bn)
        self.bn2   = nn.BatchNorm2d(channels) if bn else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        y = self.bn2(y)
        out = F.relu(x + y, inplace=True)
        return out

class TTTNet(nn.Module):
    def __init__(self, N, fan_in, channels, n_res):
        super().__init__()
        self.N = N
        self.action_dim = N * N

        # Residual trunk
        self.stem = nn.Conv2d(fan_in, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.res = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_res)])

        # Policy head 
        self.p_conv = nn.Conv2d(channels, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.p_bn   = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * N * N, self.action_dim)

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.v_bn   = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(1 * N * N, 64)
        self.v_fc2  = nn.Linear(64, 1)

        # He init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x :B, C, N, N) -> policy logits: (B, N*N), value: (B, 1) in [-1, 1]
        """
        h = self.stem(x)
        h = self.bn(h)
        h = F.relu(h, inplace=True)
        h = self.res(h)

        # Policy
        p = self.p_conv(h)
        p = self.p_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(p.size(0), -1)
        policy_logits = self.p_fc(p)

        # Value
        v = self.v_conv(h)
        v = self.v_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v), inplace=True)
        value = torch.tanh(self.v_fc2(v))

        return policy_logits, value