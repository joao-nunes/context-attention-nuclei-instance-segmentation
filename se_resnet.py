import torch
from torchvision.ops.misc import FrozenBatchNorm2d
from torch import Tensor
# ResNet-50 Functions and Classes
# Helper Function (from:
# https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18)
from typing import Optional, Callable


def conv3x3(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1) -> torch.nn.Conv2d:
    """3x3 convolution with padding"""

    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


# Helper Function (from:
# https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18)
def conv1x1(
        in_planes: int,
        out_planes: int,
        stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""

    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False)


# Squeeze-Excitation Layer (from: https://github.com/moskomule/senet.pytorch)
class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        # Average Pooling Layer
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        # FC Layer
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    # Method: forward

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# SE Bottleneck Layer for ResNet-50 (from:
# https://github.com/moskomule/senet.pytorch)
class SEBottleneck(torch.nn.Module):

    # Object attribute
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
            *,
            reduction=16):
        super(SEBottleneck, self).__init__()

        # Init variables
        # self.expansion = 4

        # Conv + BN 1
        self.conv1 = torch.nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(planes, eps=0.0)

        # Conv + BN 2
        self.conv2 = torch.nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = FrozenBatchNorm2d(planes, eps=0.0)# torch.nn.BatchNorm2d(planes)

        # Conv + BN 3
        self.conv3 = torch.nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 =  FrozenBatchNorm2d(planes * 4, eps=0.0)

        # ReLU
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.relu3 = torch.nn.ReLU(inplace=False)

        # Squeeze-Excitation Block
        self.se = SELayer(planes * 4, reduction)

        # Downsample
        self.downsample = downsample

        # Stride
        self.stride = stride

    # Method: forward

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNetBottleneck(torch.nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = FrozenBatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, eps=0.0)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, eps=0.0)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, eps=0.0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
