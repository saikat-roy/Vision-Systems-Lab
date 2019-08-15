import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            self.downsample = conv1x1(in_channels, out_channels, stride=2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out = out+x
        return out



class Resnet18(nn.Module):
    def __init__(self, input_channels):
        super(Resnet18, self).__init__()
        # Layer 1
        self.block1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2))

        # Layer 2
        self.block2 = nn.Sequential(BasicBlock(64, 64, downsample=True),
                                    BasicBlock(64, 64, downsample=False))
        # Layer 3
        self.block3 = nn.Sequential(BasicBlock(64, 128, downsample=True),
                                    BasicBlock(128, 128, downsample=False))

        # Layer 4
        self.block4 = nn.Sequential(BasicBlock(128, 256, downsample=True),
                                    BasicBlock(256, 256, downsample=False))

        # Layer 5
        self.block2 = nn.Sequential(BasicBlock(256, 512, downsample=True),
                                    BasicBlock(512, 512, downsample=False))

    def forward(self, x):

        out1 = self.block1(x)
        out1 = self.block2(out1)
        out2 = self.block3(out1)
        out3 = self.block4(out2)
        out4 = self.block5(out3)

        return out1, out2, out3, out4


class DecoderBlock(nn.Module):

    def __init__(self):
        super(DecoderBlock, self).__init__()
        


class NimbroNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.res1 = conv1x1(64, 128)
        self.res2 = conv1x1(128, 128)
        self.res3 = conv1x1(256, 128)
        # self.upsample1 =

    # def forward(self, x):
