import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.densenet import DenseNet

def convtrans3x3(in_channels, out_channels, stride=2, padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=padding)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

class Resnet18Encoder(ResNet):

    def __init__(self, freeze=False, *args, **kwargs):
        # Initializing Resnet18 using similar initializations as torch model zoo
        super(Resnet18Encoder, self).__init__(BasicBlock, [2, 2, 2, 2], *args, **kwargs)

        #print(list(self.children()))
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

class Resnet18Decoder(nn.Module):

    def __init__(self):
        super(Resnet18Decoder, self).__init__()
        self.convtrans4 = convtrans3x3(512, 128)
        self.convtrans3 = convtrans3x3(256, 128)
        self.convtrans2 = convtrans3x3(256, 128)
        self.convtrans1 = convtrans3x3(256, 4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, x1, x2, x3, x4):

        x = self.convtrans4(F.relu(x4, inplace=True))
        x3 = F.pad(x3, (1, 0, 1, 0, 0, 0, 0, 0), mode='constant', value=0)
#         print(x.size(), x3.size())
        x = torch.cat((x, x3), dim=1)
        #print(x.size())
        # NOT sure if relu should go first or bn here. Fig says relu
        x = self.convtrans3(self.bn3(F.relu(x)))
        x2 = F.pad(x2, (1, 2, 1, 2, 0, 0, 0, 0), mode='constant', value=0)
#         print(x.size(), x2.size())
        x = torch.cat((x, x2), dim=1)
        #print(x.size())
        x = self.convtrans2(self.bn2(F.relu(x)))
#         x1 = F.pad(x1, (3, 4, 4, 3, 0, 0, 0, 0), mode='constant', value=0)
        x = x[:,:,3:123,3:163]
        print(x.size(), x1.size())
        x = torch.cat((x, x1), dim=1)
        #print(x.size())
        x = self.convtrans1(self.bn1(F.relu(x)))

        #print(x.size())
        return x

class Resnet18NimbroNet(nn.Module):

    def __init__(self):
        super(Resnet18NimbroNet, self).__init__()
        self.res1 = conv1x1(64, 128)
        self.res2 = conv1x1(128, 128)
        self.res3 = conv1x1(256, 128)
        self.encoder = Resnet18Encoder(freeze=True)
        self.decoder = Resnet18Decoder()

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1 = self.res1(x1)
        x2 = self.res2(x2)
        x3 = self.res3(x3)
        x = self.decoder(x1, x2, x3, x4)
        return x


class DenseNet121Encoder(DenseNet):

    def __init__(self, freeze=False, *args, **kwargs):
        # Initializing Resnet18 using similar initializations as torch model zoo
        super(DenseNet121Encoder, self).__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                                                 *args, **kwargs)

        # print(list(self.children()))
        # Loading from pretrained model
        # print(list(torch.load('./densenet121-a639ec97.pth')))
        self._load_state_dict('./densenet121-a639ec97.pth')

        #         a = self.features
        #         print(type(a), a)
        self._create_separate_blocks()

        # Deleting unncecessary layers
        del self.features
        del self.classifier

        # Can freeze training if needed
        if freeze:
            self.freeze()

    def _create_separate_blocks(self):

        self.block1 = self.features[0:4]
        self.block2 = self.features[4:6]
        self.block3 = self.features[6:8]
        self.block4 = self.features[8:10]
        self.block5 = self.features[10:12]

        # print(self.block1, self.block2, self.block3, self.block4, self.block5)

    def _load_state_dict(self, path):
        # REDEFINING PRIVATE FUNCTION FROM PYTORCH AS CLASS METHOD
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        state_dict = torch.load(path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.load_state_dict(state_dict)

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
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        print(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x1, x2, x3, x4, x5


class DenseNetDecoder(nn.Module):

    def __init__(self):
        super(DenseNetDecoder, self).__init__()
        self.convtrans5 = convtrans3x3(1024, 128, stride=1)
        self.convtrans4 = convtrans3x3(256, 128)
        self.convtrans3 = convtrans3x3(256, 128)
        self.convtrans2 = convtrans3x3(256, 64)
        self.conv1 = conv1x1(128, 4, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.convtrans5(F.relu(x5, inplace=True))
        x4 = F.pad(x4, (1, 1, 1, 1, 0, 0, 0, 0), mode='constant', value=0)
        #         print(x.size(), x4.size())
        x = torch.cat((x, x4), dim=1)

        x = self.convtrans4(F.relu(x, inplace=True))
        x3 = F.pad(x3, (2, 3, 2, 3, 0, 0, 0, 0), mode='constant', value=0)
        #         print(x.size(), x3.size())
        x = torch.cat((x, x3), dim=1)
        # print(x.size())
        # NOT sure if relu should go first or bn here. Fig says relu
        x = self.convtrans3(self.bn3(F.relu(x)))
        x2 = F.pad(x2, (5, 6, 5, 6, 0, 0, 0, 0), mode='constant', value=0)
        #         print(x.size(), x2.size())
        x = torch.cat((x, x2), dim=1)
        #         print(x.size())
        x = self.convtrans2(self.bn2(F.relu(x)))
        #         x1 = F.pad(x1, (3, 4, 6, 5, 0, 0, 0, 0), mode='constant', value=0)
        x = x[:, :, 23:168, 13:173]
        print(x.size(), x1.size())
        x = torch.cat((x, x1), dim=1)
        # print(x.size())
        x = self.conv1(self.bn1(F.relu(x)))

        # print(x.size())
        return x


class DenseNimbroNet(nn.Module):

    def __init__(self):
        super(DenseNimbroNet, self).__init__()
        self.res3 = conv1x1(256, 128)
        self.res4 = conv1x1(512, 128)
        self.encoder = DenseNet121Encoder(freeze=True)
        self.decoder = DenseNetDecoder()

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x4 = self.res4(x4)
        x3 = self.res3(x3)
        x = self.decoder(x1, x2, x3, x4, x5)
        return x


if __name__ == "__main__":
    m = Resnet18NimbroNet()

