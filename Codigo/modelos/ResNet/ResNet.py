import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Bottleneck(nn.Module):
    """
    Bloque bottleneck usado en ResNet-50 y ResNet-101.
    """

    expansion = 4

    def __init__(self, in_planes, planes, downsample=None, stride=1):
        """
        Inicializa un bloque bottleneck.

        Args:
            in_planes (int): número de canales de entrada.
            planes (int): número de canales base del bloque.
            downsample (nn.Module | None): módulo para adaptar la conexión residual.
            stride (int): stride aplicado en la convolución 3x3.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): tensor de entrada con forma [batch, channels, height, width].

        Returns:
            Tensor: salida del bloque tras aplicar conexión residual.
        """
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
    """
    Implementación simplificada de ResNet para Tiny ImageNet.
    """

    def __init__(self, block, layers, num_classes=200):
        """
        Inicializa ResNet.

        Args:
            block (nn.Module): tipo de bloque residual usado.
            layers (list[int]): número de bloques por etapa.
            num_classes (int): número de clases de salida.
        """
        super().__init__()

        self.in_planes = 64
        self.use_checkpointing = False

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = self._make_layer(
            block=block,
            planes=64,
            blocks=layers[0]
        )

        self.layer2 = self._make_layer(
            block=block,
            planes=128,
            blocks=layers[1],
            stride=2
        )

        self.layer3 = self._make_layer(
            block=block,
            planes=256,
            blocks=layers[2],
            stride=2
        )

        self.layer4 = self._make_layer(
            block=block,
            planes=512,
            blocks=layers[3],
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): imágenes con forma [batch, 3, height, width].

        Returns:
            Tensor: logits con forma [batch, num_classes].
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.use_checkpointing and self.training:
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Construye una etapa de ResNet formada por varios bloques residuales.

        Args:
            block (nn.Module): tipo de bloque residual.
            planes (int): número de canales base de la etapa.
            blocks (int): número de bloques en la etapa.
            stride (int): stride del primer bloque de la etapa.

        Returns:
            nn.Sequential: secuencia de bloques residuales.
        """
        downsample = None
        layers = []

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                downsample=downsample,
                stride=stride
            )
        )

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes
                )
            )

        return nn.Sequential(*layers)


def ResNet50(num_classes=200):
    """
    Construye una ResNet-50.

    Args:
        num_classes (int): número de clases de salida.

    Returns:
        ResNet: modelo ResNet-50.
    """
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes
    )


def ResNet101(num_classes=200):
    """
    Construye una ResNet-101.

    Args:
        num_classes (int): número de clases de salida.

    Returns:
        ResNet: modelo ResNet-101.
    """
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes
    )