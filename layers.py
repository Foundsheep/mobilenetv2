import torch
from torch import nn
from torch.nn import functional as F


class BottleneckResidualBlock(nn.Module):

    def __init__(self, first_channel, last_channel, factor, stride):
        super().__init__()
        self.stride = stride
        self.conv_1 = nn.Conv2d(in_channels=first_channel, out_channels=int(first_channel*factor), kernel_size=1)

        # channel to be used in the block
        c = self.conv_1.out_channels
        self.bn_1 = nn.BatchNorm2d(c)

        self.conv_2_dw = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=stride, groups=c, padding='same')
        self.bn_2 = nn.BatchNorm2d(c)

        self.conv_2_pw = nn.Conv2d(in_channels=c, out_channels=last_channel, kernel_size=1)
        self.bn_3 = nn.BatchNorm2d(last_channel)

    def forward(self, inputs):
        identity = nn.Identity()(inputs)
        x = F.relu(self.bn_1(self.conv_1(inputs)))
        x = F.relu(self.bn_2(self.conv_2_dw(x)))
        x = self.bn_3(self.conv_2_pw(x))

        if self.stride == 1 and identity.size() == x.size():
            x += identity
        return x


def test():
    bottleneck_1 = BottleneckResidualBlock(first_channel=16, last_channel=32, factor=4, stride=1)
    x = torch.randn((100, 16, 224, 224))
    output = bottleneck_1(x)
    print(output.size())

    bottleneck_2 = BottleneckResidualBlock(first_channel=32, last_channel=32, factor=4, stride=1)
    output2 = bottleneck_2(output)
    print(output2.size())


if __name__ == "__main__":
    test()
