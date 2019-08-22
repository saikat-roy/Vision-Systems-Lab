import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock


def convtrans3x3(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)


class Resnet18Encoder(ResNet):

    def __init__(self, freeze=False, *args, **kwargs):
        # Initializing Resnet18 using similar initializations as torch model zoo
        super(Resnet18Encoder, self).__init__(BasicBlock, [2, 2, 2, 2], *args, **kwargs)

        print(list(self.children()))
        # Loading from pretrained model
        self.load_state_dict(torch.load('./resnet18-5c106cde.pth'))

        # Deleting unncessary layers
        del self.avgpool
        del self.fc

        # Can freeze training if needed
        if freeze:
            self.freeze()

    def freeze(self):
        # Freezes the weights of the entire encoder
        for child in self.children():
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze(self):
        # Unfreezes the weights of the encoder
        for child in self.children():
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x):
        # Overriding the forward method
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x1, x2, x3, x4


class DecoderBlock(nn.Module):

    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.convtrans4 = convtrans3x3(512, 128)
        self.convtrans3 = convtrans3x3(256, 128)
        self.convtrans2 = convtrans3x3(256, 128)
        self.convtrans1 = convtrans3x3(256, 4, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, x1, x2, x3, x4):

        x = self.convtrans4(F.relu(x4, inplace=True))
        #print(x.size())
        x3 = F.pad(x3, (0, 1, 1, 1, 0, 0, 0, 0), mode='constant', value=0)
        x = torch.cat((x, x3), dim=1)
        #print(x.size())
        # NOT sure if relu should go first or bn here. Fig says relu
        x = self.convtrans3(self.bn3(F.relu(x)))
        x2 = F.pad(x2, (1, 2, 2, 3, 0, 0, 0, 0), mode='constant', value=0)
        x = torch.cat((x, x2), dim=1)
        #print(x.size())
        x = self.convtrans2(self.bn2(F.relu(x)))
        x1 = F.pad(x1, (3, 4, 6, 5, 0, 0, 0, 0), mode='constant', value=0)
        x = torch.cat((x, x1), dim=1)
        #print(x.size())
        x = self.convtrans1(self.bn1(F.relu(x)))

        #print(x.size())
        return x


class NimbroNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(NimbroNet, self).__init__()
        self.res1 = conv1x1(64, 128)
        self.res2 = conv1x1(128, 128)
        self.res3 = conv1x1(256, 128)
        self.encoder = Resnet18Encoder(freeze=True)
        self.decoder = DecoderBlock()

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1 = self.res1(x1)
        x2 = self.res2(x2)
        x3 = self.res3(x3)
        x = self.decoder(x1, x2, x3, x4)
        return x
