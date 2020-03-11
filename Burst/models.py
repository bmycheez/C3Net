import torch.nn as nn
from math import sqrt
import torch


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        out = self.dncnn(x)
        return out


class GlobalMaxPool(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, input):
        output = torch.max(input, dim=1)[0]

        return torch.unsqueeze(output, 1)


class CA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RB(nn.Module):
    def __init__(self, features):
        super(RB, self).__init__()
        layers = []
        kernel_size = 3
        for _ in range(1):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features
                                    , kernel_size=kernel_size, padding=kernel_size//2, bias=True))
            layers.append(nn.PReLU())
            layers.append(nn.Conv2d(in_channels=features, out_channels=features
                                    , kernel_size=kernel_size, padding=kernel_size//2, bias=True))
        self.res = nn.Sequential(*layers)
        self.ca = CA(features)

    def forward(self, x):
        out = self.res(x)
        out = self.ca(out)
        out += x
        return out


class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()

        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2 * channel_in, kernel_size=3, stride=2, padding=1)
        self.relu = nn.PReLU()

    def forward(self, x):

        out = self.relu(self.conv(x))

        return out


class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        self.conv = nn.PixelShuffle(2)
        self.relu = nn.PReLU()

    def forward(self, x):

        out = self.relu(self.conv(x))

        return out


class AB(nn.Module):
    def __init__(self, features):
        super(AB, self).__init__()

        num = 2
        self.DCR_block1 = self.make_layer(RB, features, num)
        self.down1 = self.make_layer(_down, features, 1)
        self.DCR_block2 = self.make_layer(RB, features * 2, num)
        self.down2 = self.make_layer(_down, features * 2, 1)
        self.DCR_block3 = self.make_layer(RB, features * 4, num)
        self.up2 = self.make_layer(_up, features * 8, 1)
        self.DCR_block22 = self.make_layer(RB, features * 4, num)
        self.up1 = self.make_layer(_up, features * 4, 1)
        self.DCR_block11 = self.make_layer(RB, features * 2, num)
        self.conv_f = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in, num):
        layers = []
        for _ in range(num):
            layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):

        conc1 = self.DCR_block1(x)
        out = self.down1(conc1)

        conc2 = self.DCR_block2(out)
        conc3 = self.down2(conc2)

        out = self.DCR_block3(conc3)
        out = torch.cat([conc3, out], 1)

        out = self.up2(out)
        out = torch.cat([conc2, out], 1)
        out = self.DCR_block22(out)

        out = self.up1(out)
        out = torch.cat([conc1, out], 1)
        out = self.DCR_block11(out)

        out = self.relu2(self.conv_f(out))
        out += x

        return out


class GAB(nn.Module):
    def __init__(self, features):
        super(GAB, self).__init__()

        kernel_size = 3
        self.res1 = self.make_layer(RB, features, 2)
        self.R = 2
        self.A = 1
        self.RB = nn.ModuleList()
        for _ in range(self.R):
            self.RB.append(RB(features))
        self.AB = nn.ModuleList()
        for _ in range(self.A):
            self.AB.append(AB(features))
        self.GFF_R = nn.Sequential(
            nn.Conv2d(self.R * features, features, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(features, features, kernel_size, padding=kernel_size//2, stride=1),
        )
        self.GFF_A = nn.Sequential(
            nn.Conv2d(self.A * features, features, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(features, features, kernel_size, padding=kernel_size//2, stride=1),
        )
        self.softmax = nn.Sigmoid()
        self.res2 = self.make_layer(RB, features * 2, 2)
        self.conv_f = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in, num):
        layers = []
        for _ in range(num):
            layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.res1(x)

        RB_outs = []
        for i in range(self.R):
            outR = self.RB[i](out)
            RB_outs.append(outR)
        outR = torch.cat(RB_outs, 1)
        outR = self.GFF_R(outR)

        AB_outs = []
        for i in range(self.A):
            outA = self.AB[i](out)
            AB_outs.append(outA)
        outA = torch.cat(AB_outs, 1)
        outA = self.GFF_A(outA)

        # outR *= self.softmax(outA)
        out = torch.cat([outR, outA], 1)
        out = self.relu2(self.conv_f(self.res2(out)))
        out += x

        return out


class Net(nn.Module):
    def __init__(self, features=48):
        super(Net, self).__init__()

        kernel_size = 3
        self.conv_i = nn.Conv2d(in_channels=3, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.GA = 5
        self.maxpool = GlobalMaxPool()
        self.GAB = nn.ModuleList()
        for _ in range(self.GA):
            self.GAB.append(GAB(features))
        self.conv_m = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.relum = nn.PReLU()
        self.GFF_GA = nn.Sequential(
            nn.Conv2d(self.GA * 1 * features * 7, features, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(features, features, kernel_size, padding=kernel_size//2, stride=1),
        )
        self.conv_f = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in, num):
        layers = []
        for _ in range(num):
            layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, im, c, h, w = x.size()
        out = self.relu1(self.conv_i(x.view((b*im, c, h, w))))

        GAB_outs = []
        for i in range(self.GA):
            out = self.GAB[i](out)
            out_max = self.maxpool(out.view((b, im, -1, h, w)))
            out_max = out_max.repeat(1, im, 1, 1, 1).view(b*im, -1, h, w)
            out = self.relum(self.conv_m(torch.cat([out, out_max], 1)))
            GAB_outs.append(out)
        out = torch.cat(GAB_outs, 1)
        out = self.GFF_GA(out.view((b, -1, h, w)))

        out = self.relu2(self.conv_f(out))
        out += x[:, 3, :, :, :]

        return out
