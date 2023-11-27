import torch
from torch import nn
from torchvision import transforms
from torchsummary import summary

from typing import Union


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3):
        super().__init__()
        # --- reference ---
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # -----------------
        self.depthwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1, groups=in_channels, stride=stride)
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = nn.ReLU()
        self.pointwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        x = self.depthwise_layer(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.pointwise_layer(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, width_multiplier=None, in_channels=3, out_channels=32, kernel_size=3, stride=2):
        super(ConvBlock, self).__init__()
        if width_multiplier:
            self.out_channels = int(32 * width_multiplier)
        else:
            self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=self.out_channels,
                              padding=1,
                              kernel_size=self.kernel_size,
                              stride=self.stride)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, width_multiplier: Union[int, float] = 1,
                 resolution_multiplier: Union[int, float] = 1,
                 is_mobile: bool = True,
                 num_classes: int = 1000):
        super(MobileNet, self).__init__()
        self.resolution_multiplier = resolution_multiplier

        base_conv = DepthwiseSeparableConvolution if is_mobile else ConvBlock
        kernel_size = 3
        self.conv_01 = ConvBlock(width_multiplier)
        self.dp_layer_02 = base_conv(in_channels=self.conv_01.out_channels, out_channels=self.conv_01.out_channels * 2, stride=1, kernel_size=kernel_size)
        self.dp_layer_03 = base_conv(in_channels=self.dp_layer_02.out_channels, out_channels=self.dp_layer_02.out_channels * 2, stride=2, kernel_size=kernel_size)
        self.dp_layer_04 = base_conv(in_channels=self.dp_layer_03.out_channels, out_channels=self.dp_layer_03.out_channels, stride=1, kernel_size=kernel_size)
        self.dp_layer_05 = base_conv(in_channels=self.dp_layer_04.out_channels, out_channels=self.dp_layer_04.out_channels * 2, stride=2, kernel_size=kernel_size)
        self.dp_layer_06 = base_conv(in_channels=self.dp_layer_05.out_channels, out_channels=self.dp_layer_05.out_channels, stride=1, kernel_size=kernel_size)
        self.dp_layer_07 = base_conv(in_channels=self.dp_layer_06.out_channels, out_channels=self.dp_layer_06.out_channels * 2, stride=2, kernel_size=kernel_size)
        self.dp_layer_list = nn.ModuleList(
            [base_conv(in_channels=self.dp_layer_07.out_channels, out_channels=self.dp_layer_07.out_channels, stride=1, kernel_size=kernel_size) for _ in range(5)]
        )
        self.dp_layer_13 = base_conv(in_channels=self.dp_layer_07.out_channels, out_channels=self.dp_layer_07.out_channels * 2, stride=2, kernel_size=kernel_size)

        # the last layer is written to have stride 2, but given the fact the shape doesn't shrink, I've put it 1
        self.dp_layer_14 = base_conv(in_channels=self.dp_layer_13.out_channels, out_channels=self.dp_layer_13.out_channels, stride=1, kernel_size=kernel_size)
        self.ga = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=self.dp_layer_14.out_channels, out_features=num_classes)

    def forward(self, x):
        if self.resolution_multiplier != 1:
            B, C, H, W = x.size()
            x = transforms.Resize(int(H*self.resolution_multiplier))(x)
        x = self.conv_01(x)
        x = self.dp_layer_02(x)
        x = self.dp_layer_03(x)
        x = self.dp_layer_04(x)
        x = self.dp_layer_05(x)
        x = self.dp_layer_06(x)
        x = self.dp_layer_07(x)
        for layer in self.dp_layer_list:
            x = layer(x)
        x = self.dp_layer_13(x)
        x = self.dp_layer_14(x)
        x = self.ga(x)
        B, C, H, W = x.size()
        x = x.view(B, -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":

    model = MobileNet(is_mobile=True)
    summary(model, (3, 224, 224))
    inp = torch.randn(100, 3, 224, 224)
    output = model(inp)
    print(output.size())