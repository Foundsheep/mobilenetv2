import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class BottleneckResidualBlock(nn.Module):

    def __init__(self, first_channel, last_channel, factor, stride):
        super().__init__()
        self.stride = stride
        self.conv_1 = nn.Conv2d(in_channels=first_channel, out_channels=int(first_channel*factor), kernel_size=1, stride=1)

        # channel to be used in the block
        c = self.conv_1.out_channels
        self.bn_1 = nn.BatchNorm2d(c)

        self.conv_2_dw = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=stride, groups=c, padding=1)
        self.bn_2 = nn.BatchNorm2d(c)

        self.conv_2_pw = nn.Conv2d(in_channels=c, out_channels=last_channel, kernel_size=1)
        self.bn_3 = nn.BatchNorm2d(last_channel)
        self.last_channel = last_channel

    def forward(self, inputs):
        identity = nn.Identity()(inputs)
        x = F.relu6(self.bn_1(self.conv_1(inputs)))
        x = F.relu6(self.bn_2(self.conv_2_dw(x)))
        x = self.bn_3(self.conv_2_pw(x))

        if self.stride == 1 and identity.size() == x.size():
            x += identity
        return x


class MobileNetV2(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.layer_1_conv = nn.Conv2d(in_channels=3, out_channels=32, stride=2, kernel_size=3, padding=1)
        self.layer_2_bottleneck = BottleneckResidualBlock(first_channel=self.layer_1_conv.out_channels, last_channel=16, factor=1, stride=1)

        tmp_lst = []
        for i in range(2):
            if i == 0:
                tmp_lst.append(BottleneckResidualBlock(first_channel=self.layer_2_bottleneck.last_channel, last_channel=24, factor=6, stride=2))
            else:
                tmp_lst.append(BottleneckResidualBlock(first_channel=24, last_channel=24, factor=6, stride=1))
        self.layer_3_bottleneck = nn.Sequential(*tmp_lst)

        tmp_lst = []
        for i in range(3):
            if i == 0:
                tmp_lst.append(BottleneckResidualBlock(first_channel=self.layer_3_bottleneck[-1].last_channel, last_channel=32, factor=6, stride=2))
            else:
                tmp_lst.append(BottleneckResidualBlock(first_channel=32, last_channel=32, factor=6, stride=1))
        self.layer_4_bottleneck = nn.Sequential(*tmp_lst)

        tmp_lst = []
        for i in range(4):
            if i == 0:
                tmp_lst.append(BottleneckResidualBlock(first_channel=self.layer_4_bottleneck[-1].last_channel, last_channel=64, factor=6, stride=2))
            else:
                tmp_lst.append(BottleneckResidualBlock(first_channel=64, last_channel=64, factor=6, stride=1))
        self.layer_5_bottleneck = nn.Sequential(*tmp_lst)

        tmp_lst = []
        for i in range(3):
            if i == 0:
                tmp_lst.append(BottleneckResidualBlock(first_channel=self.layer_5_bottleneck[-1].last_channel, last_channel=96, factor=6, stride=1))
            else:
                tmp_lst.append(BottleneckResidualBlock(first_channel=96, last_channel=96, factor=6, stride=1))
        self.layer_6_bottleneck = nn.Sequential(*tmp_lst)

        tmp_lst = []
        for i in range(3):
            if i == 0:
                tmp_lst.append(BottleneckResidualBlock(first_channel=self.layer_6_bottleneck[-1].last_channel, last_channel=160, factor=6, stride=2))
            else:
                tmp_lst.append(BottleneckResidualBlock(first_channel=160, last_channel=160, factor=6, stride=1))
        self.layer_7_bottleneck = nn.Sequential(*tmp_lst)

        self.layer_8_conv = nn.Conv2d(in_channels=self.layer_7_bottleneck[-1].last_channel, out_channels=1280, stride=1, kernel_size=1, padding='same')
        self.layer_9_avp = nn.AdaptiveAvgPool2d(1)
        self.layer_10_conv = nn.Conv2d(in_channels=self.layer_8_conv.out_channels, out_channels=self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.layer_1_conv(x)
        x = self.layer_2_bottleneck(x)
        x = self.layer_3_bottleneck(x)
        x = self.layer_4_bottleneck(x)
        x = self.layer_5_bottleneck(x)
        x = self.layer_6_bottleneck(x)
        x = self.layer_7_bottleneck(x)
        x = self.layer_8_conv(x)
        x = self.layer_9_avp(x)
        x = self.layer_10_conv(x)
        return x


def test():
    bottleneck_1 = BottleneckResidualBlock(first_channel=16, last_channel=32, factor=4, stride=1)
    x = torch.randn((100, 16, 224, 224))
    output = bottleneck_1(x)
    print(output.size())

    bottleneck_2 = BottleneckResidualBlock(first_channel=32, last_channel=32, factor=4, stride=1)
    output2 = bottleneck_2(output)
    print(output2.size())


def test_model():
    mobilenetv2 = MobileNetV2(class_num=10)
    summary(mobilenetv2, (3, 224, 224))


if __name__ == "__main__":
    test_model()
