import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ConvBNReLU, ReceptiveConv
from models.vgg import vgg16
from models.resnet import resnet50, resnet101, resnet152, Bottleneck
from models.MobileNetV2 import mobilenetv2


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EDN(nn.Module):
    def __init__(self, arch='mobilenetv2', pretrained=None, use_carafe=True,
                 enc_channels=[64, 128, 256, 512, 512, 256, 256],
                 dec_channels=[32, 64, 128, 128, 256, 256, 256], freeze_s1=False):
        super(EDN, self).__init__()
        
        self.arch = arch
        self.backbone = eval(arch)(pretrained)

        if arch == 'vgg16':
            enc_channels=[64, 128, 256, 512, 512, 256, 256]#, 256, 256]
        elif 'resnet50' in arch:
            enc_channels=[64, 256, 512, 1024, 2048, 1024, 1024]
            dec_channels=[32, 64, 256, 512, 512, 128, 128]
        elif 'mobilenetv2' in arch:
            enc_channels=[16, 24, 32, 96, 160, 40, 40]
            dec_channels=[16, 24, 32, 40, 40, 40, 40]

        use_dwconv = 'mobilenet' in arch
        
        if 'vgg' in arch:
            self.conv6 = nn.Sequential(nn.MaxPool2d(2,2,0),
            ConvBNReLU(enc_channels[-3], enc_channels[-2]),                                   
                                       ConvBNReLU(enc_channels[-2], enc_channels[-2], residual=False),
                                      )
            self.conv7 = nn.Sequential(nn.MaxPool2d(2,2,0),
            ConvBNReLU(enc_channels[-2], enc_channels[-1]),
                                       ConvBNReLU(enc_channels[-1], enc_channels[-1], residual=False),
                                      )
        elif 'resnet' in arch:
            self.inplanes = enc_channels[-3]
            self.base_width = 64
            self.conv6 = nn.Sequential(
                                       self._make_layer(enc_channels[-2] // 4, 2, stride=2),
                                      )
            self.conv7 = nn.Sequential(
                                       self._make_layer(enc_channels[-2] // 4, 2, stride=2),
                                      )
        elif 'mobilenet' in arch:
            self.conv6 = nn.Sequential(
                                       InvertedResidual(enc_channels[-3], enc_channels[-2], stride=2),
                                       InvertedResidual(enc_channels[-2], enc_channels[-2]),
                                      )
            self.conv7 = nn.Sequential(
                                       InvertedResidual(enc_channels[-2], enc_channels[-1], stride=2),
                                       InvertedResidual(enc_channels[-1], enc_channels[-1]),
                                      )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fpn = CustomDecoder(enc_channels, dec_channels, use_dwconv=use_dwconv)
        
        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)
        self._freeze_backbone(freeze_s1=freeze_s1)
        
    
    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=groups,
                                base_width=self.base_width, dilation=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _freeze_backbone(self, freeze_s1):
        if not freeze_s1:
            return
        assert('resnet' in self.arch and '3x3' not in self.arch)
        m = [self.backbone.conv1, self.backbone.bn1, self.backbone.relu]
        print("freeze stage 0 of resnet")
        for p in m:
            for pp in p.parameters():
                p.requires_grad = False

    def forward(self, input):
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)
        
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        attention = torch.sigmoid(self.gap(conv7))

        features = self.fpn([conv1, conv2, conv3, conv4, conv5, conv6, conv7], attention)
        
        saliency_maps = []
        for idx, feature in enumerate(features[:5]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )

        return torch.sigmoid(torch.cat(saliency_maps, dim=1))


class CustomDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_dwconv=False):
        super(CustomDecoder, self).__init__()
        self.inners_a = nn.ModuleList()
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))
        
        self.fuse = nn.ModuleList()
        dilation = [[1, 2, 4, 8]] * (len(in_channels) - 4) + [[1, 2, 3, 4]] * 2 + [[1, 1, 1, 1]] * 2
        baseWidth = [32] * (len(in_channels) - 5) + [24] * 5
        print("using dwconv:", use_dwconv)
        for i in range(len(in_channels)):
            self.fuse.append(nn.Sequential(
                    ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=use_dwconv),
                    ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=use_dwconv)))

    def forward(self, features, att=None):
        if att is not None:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1] * att))
        else:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        num_mul_att = 1
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            if att is not None and att.shape[1] == features[idx].shape[1] and num_mul_att:
                features[idx] = features[idx] * att
                num_mul_att -= 1
            inner_lateral = self.inners_a[idx](features[idx])
            stage_result = self.fuse[idx](torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results
