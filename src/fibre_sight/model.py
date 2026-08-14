'''
Created on 4 April 2026

Modified on 2 August 2026 to add optional attention gates to the skip paths
Modified on 14 August 2026

small U-Net used for channel-2 axon masks

@author: Dinghao Luo
'''

#%% imports
import torch
import torch.nn as nn


#%% blocks
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels, gate_channels):
        super().__init__()
        attention_channels = max(1, min(skip_channels, gate_channels) // 2)
        self.skip_projection = nn.Conv2d(
            skip_channels,
            attention_channels,
            kernel_size=1,
            )
        self.gate_projection = nn.Conv2d(
            gate_channels,
            attention_channels,
            kernel_size=1,
            )
        self.score = nn.Conv2d(attention_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip, gate):
        attention = self.skip_projection(skip) + self.gate_projection(gate)
        attention = self.sigmoid(self.score(self.relu(attention)))
        return skip * attention


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, attention_gates=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 2 August 2026: gate only the skip path so the working decoder stays intact
        self.attention = (
            AttentionGate(skip_channels, out_channels)
            if attention_gates else None
            )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = pad_to_match(x, skip)
        if self.attention is not None:
            skip = self.attention(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


#%% model
class SmallUNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            base_channels=24,
            depth=4,
            attention_gates=False,
            ):
        super().__init__()
        if depth < 2:
            raise ValueError('depth must be at least 2')

        # the 512-pixel trial widens the first working model in its YAML recipe
        channels = [base_channels * (2 ** idx) for idx in range(depth)]
        self.in_conv = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList([
            Down(channels[idx - 1], channels[idx])
            for idx in range(1, depth)
            ])
        self.ups = nn.ModuleList([
            Up(
                channels[idx],
                channels[idx - 1],
                channels[idx - 1],
                attention_gates=attention_gates,
                )
            for idx in range(depth - 1, 0, -1)
            ])
        self.out_conv = OutConv(channels[0], out_channels)

    def forward(self, x):
        features = [self.in_conv(x)]
        for down in self.downs:
            features.append(down(features[-1]))

        x = features[-1]
        for up, skip in zip(self.ups, reversed(features[:-1])):
            x = up(x, skip)

        return self.out_conv(x)


def build_model(config):
    return SmallUNet(**config)


def pad_to_match(x, reference):
    diff_y = reference.size(2) - x.size(2)
    diff_x = reference.size(3) - x.size(3)

    if diff_y == 0 and diff_x == 0:
        return x

    return nn.functional.pad(
        x,
        [diff_x // 2, diff_x - diff_x // 2,
         diff_y // 2, diff_y - diff_y // 2],
        )
