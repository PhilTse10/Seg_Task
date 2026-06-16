import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        return x + residual


class FeatureFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_conv_unit1 = ResidualConvUnit(channels)
        self.res_conv_unit2 = ResidualConvUnit(channels)

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.res_conv_unit1(skip)
        x = self.res_conv_unit2(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class DPTDecoder(nn.Module):
    """DPT-style decoder for multiscale channel-first encoder features."""

    def __init__(
        self,
        in_channels=(128, 256, 512, 1024, 2048),
        features=256,
        num_classes=3,
        dropout=0.1,
    ):
        super().__init__()
        self.projects = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, features, kernel_size=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False),
                )
                for in_ch in in_channels
            ]
        )
        self.refinenets = nn.ModuleList(
            [FeatureFusionBlock(features) for _ in range(len(in_channels) - 1)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != len(self.projects):
            raise ValueError(f"DPTDecoder expects {len(self.projects)} feature maps.")

        projected = [project(feature) for project, feature in zip(self.projects, features)]

        x = projected[-1]
        for idx, skip in enumerate(reversed(projected[:-1])):
            x = self.refinenets[idx](x, skip)
        return self.head(x)


class DPTDecoderHead(nn.Module):
    """DPT decoder with bilinear upsampling to the input resolution."""

    def __init__(self, in_channels, num_classes, features=256, dropout=0.1):
        super().__init__()
        self.decoder = DPTDecoder(
            in_channels=in_channels,
            features=features,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, features, output_size):
        logits = self.decoder(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
