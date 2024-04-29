import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
from torch import Tensor


# period2 主要架构：Segmentation Network

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def make_layer(block, inplanes, planes, blocks, stride=1, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, norm_layer=norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, norm_layer=norm_layer))

    return nn.Sequential(*layers)


def l2_normalize(input, dim=1, eps=1e-12):
    denom = torch.sqrt(torch.sum(input**2, dim=dim, keepdim=True))
    return input / (denom + eps)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
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

    def forward(self, x: Tensor) -> Tensor:
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


def get_norm_layer(norm: str):
    norm = {
        "BN": nn.BatchNorm2d,
        "LN": nn.LayerNorm,
    }[norm.upper()]
    return norm


def get_act_layer(act: str):
    act = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
    }[act.lower()]
    return act


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        conv_kwargs=None,
        norm_layer=None,
        norm_kwargs=None,
        act_layer=None,
        act_kwargs=None,
    ):
        super(ConvNormAct2d, self).__init__()

        conv_kwargs = {}
        if norm_layer:
            conv_kwargs["bias"] = False
        if padding == "same" and stride > 1:
            # if kernel_size is even, -1 is must
            padding = (kernel_size - 1) // 2

        self.conv = self._build_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_kwargs,
        )
        self.norm = None
        if norm_layer:
            norm_kwargs = {}
            self.norm = get_norm_layer(norm_layer)(
                num_features=out_channels, **norm_kwargs
            )
        self.act = None
        if act_layer:
            act_kwargs = {}
            self.act = get_act_layer(act_layer)(**act_kwargs)

    def _build_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        conv_kwargs,
    ):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **conv_kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvNormAct2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    norm_layer="BN",
                    act_layer="RELU",
                ),
            )
        )
        for atrous_rate in atrous_rates:
            conv_norm_act = ConvNormAct2d
            modules.append(
                conv_norm_act(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1 if atrous_rate == 1 else 3,
                    padding=0 if atrous_rate == 1 else atrous_rate,
                    dilation=atrous_rate,
                    norm_layer="BN",
                    act_layer="RELU",
                )
            )

        self.aspp_feature_extractors = nn.ModuleList(modules)
        self.aspp_fusion_layer = ConvNormAct2d(
            (1 + len(atrous_rates)) * output_channels,
            output_channels,
            kernel_size=3,
            norm_layer="BN",
            act_layer="RELU",
        )

    def forward(self, x):
        res = []
        for aspp_feature_extractor in self.aspp_feature_extractors:
            res.append(aspp_feature_extractor(x))
        res[0] = F.interpolate(
            input=res[0], size=x.shape[2:], mode="bilinear", align_corners=False
        )  # resize back after global-avg-pooling layer
        res = torch.cat(res, dim=1)
        res = self.aspp_fusion_layer(res)
        return res


# 448-256-256
class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


class Segmentation(nn.Module):
    def __init__(self, encoder, proj_layer, bn, decoder, outputS):
        super().__init__()
        self.encoder = encoder
        self.proj_layer = proj_layer
        self.bn = bn
        self.decoder = decoder
        self.outputS = outputS
        self.segmentation_net = SegmentationNet(inplanes=448)

    def forward(self, img):
        self.encoder.eval()
        self.proj_layer.eval()
        self.bn.eval()
        self.decoder.eval()
        self.outputS.eval()


        inputs = self.encoder(img)
        features = self.proj_layer(inputs)
        outputs = self.decoder(self.bn(features))

        # teacher的输出正则化
        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.encoder(img)
        ]
        # student的输出正则化(OutputS为ReviewKD，得到融合后的新student feature)
        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.outputS(outputs)
        ]
        # 三张图叠加，size为teacher[0]
        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )

        output_segmentation = self.segmentation_net(output)

        return output_segmentation
