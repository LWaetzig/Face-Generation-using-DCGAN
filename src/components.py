import torch
import torch.nn as nn
import torch.optim as optim


class Discriminator(nn.Module):
    def __init__(
        self, img_channels: int, features: int, learning_rate: float = 2e-4
    ) -> None:
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            self.conv_block(img_channels, features, kernel_size=4, stride=2, padding=1),
            self.conv_block(features, features * 2, kernel_size=4, stride=2, padding=1),
            self.conv_block(
                features * 2, features * 4, kernel_size=4, stride=2, padding=1
            ),
            self.conv_block(
                features * 4, features * 8, kernel_size=4, stride=2, padding=1
            ),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)

    def conv_block(
        self,
        in_channles: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Sequential:
        """create a convolutional block with batch normalization and leaky relu activation

        Args:
            in_channles (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int, optional): kernel window size. Defaults to 4.
            stride (int, optional): size of striding window. Defaults to 2.
            padding (int, optional): size of padding window. Defaults to 1.

        Returns:
            nn.Sequential: a sequential block of convolutional layer, batch normalization and leaky relu activation
        """
        layer = [
            nn.Conv2d(
                in_channles,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2),
        ]
        return nn.Sequential(*layer)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)


class Generator(nn.Module):
    def __init__(
        self, z_dim: int, img_channels: int, features: int, learning_rate: float = 2e-4
    ) -> None:
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            self.trans_conv_block(
                z_dim, features * 16, kernel_size=4, stride=1, padding=0
            ),
            self.trans_conv_block(
                features * 16, features * 8, kernel_size=4, stride=2, padding=1
            ),
            self.trans_conv_block(
                features * 8, features * 4, kernel_size=4, stride=2, padding=1
            ),
            self.trans_conv_block(
                features * 4, features * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ConvTranspose2d(
                features * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)

    def trans_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Sequential:
        """creating a transposed convolutional block consisting of convolutional layer, batch normalization, relu activation and dropout layer

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            kernel_size (int, optional): kernel window size. Defaults to 4.
            stride (int, optional): size of striding window. Defaults to 2.
            padding (int, optional): size of padding window. Defaults to 1.

        Returns:
            nn.Sequential: a sequential block of transposed convolutional layer, batch normalization, relu activation and dropout layer
        """
        layer = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(0.25),
        ]
        return nn.Sequential(*layer)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
