import torch
import torch.nn as nn
from models.utils import FrozenBatchNorm2d
from models.utils import ConvBNReLU
from torchvision.models import resnet

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, first_stride=2,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=first_stride, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)
        p1 = self.maxpool(c1)

        c2 = self.layer1(p1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c1, c2, c3, c4, c5

def resnet18(pretrained=None, **kwargs):
    r"""ResNet-18 model from
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], norm_layer=FrozenBatchNorm2d, **kwargs)
    if pretrained is not None:
        print("resnet18: loading pretrained model: resnet18-5c106cde.pth")
        model.load_state_dict(torch.load("pretrained/resnet18-5c106cde.pth"), strict=False)
    return model

def resnet34(pretrained=None, **kwargs):
    r"""ResNet-34 model from
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=FrozenBatchNorm2d, **kwargs)
    if pretrained is not None:
        print("resnet34: loading pretrained model: {}".format(pretrained))
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model

def resnet50(pretrained=None, **kwargs):
    r"""ResNet-50 model from
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=FrozenBatchNorm2d, **kwargs)
    if pretrained is not None:
        print("resnet50: loading pretrained model: resnet50-19c8e357.pth")
        model.load_state_dict(torch.load("pretrained/resnet50-19c8e357.pth"), strict=False)
    return model

def resnet50_nofreeze(pretrained=None, **kwargs):
    r"""ResNet-50 model from
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=nn.BatchNorm2d, **kwargs)
    if pretrained is not None:
        print("resnet50: loading pretrained model: resnet50-19c8e357.pth")
        model.load_state_dict(torch.load("pretrained/resnet50-19c8e357.pth"), strict=False)
    return model

def resnet50_stride1(pretrained=None, **kwargs):
    r"""ResNet-50 model from
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=FrozenBatchNorm2d, first_stride=1, **kwargs)
    if pretrained is not None:
        print("resnet50: loading pretrained model: resnet50-19c8e357.pth")
        print("first conv setting:", model.conv1)
        model.load_state_dict(torch.load("resnet50-19c8e357.pth"), strict=False)
    return model

def resnet101(pretrained=None, **kwargs):
    r"""ResNet-101 model from
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=FrozenBatchNorm2d, **kwargs)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def resnet152(pretrained=None, **kwargs):
    r"""ResNet-152 model from
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], norm_layer=FrozenBatchNorm2d, **kwargs)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model
