import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Residual blocks
# -----------------------------


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.Identity()

        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=False)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.projection is not None:
            identity = self.projection(identity)

        out = out + identity
        out = F.relu(out, inplace=False)

        return out


class ResidualBlockNoProjection(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.bn1 = nn.Identity()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.bn2 = nn.Identity()

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=False)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out, inplace=False)

        return out


# -----------------------------
# Main model
# -----------------------------


class ResNet8TinyMLPerf(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.Identity()

        self.block1 = ResidualBlockNoProjection(16)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.block3 = ResidualBlock(32, 64, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=8)

        self.fc = nn.Linear(64, num_classes)

        # For structural eqivalence with tflite
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.softmax(x)

        return x

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


# -----------------------------
# Required API
# -----------------------------

ModelUnderTest = ResNet8TinyMLPerf()

ModelInputs = (torch.ones(1, 3, 32, 32),)

"""
How to convert to tflite:

```python
import resnet8_fp32
import litert_torch
edge_model = litert_torch.convert(resnet8_fp32.ModelUnderTest.eval(), resnet8_fp32.ModelInputs)
edge_model.export('resnet8_fp32_new.tflite')
```

"""
