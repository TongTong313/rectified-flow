import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """MiniUnet的下采样模块
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

        # 降采样
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = x + res

        x = self.pool(x)

        return x


class UpBlock(nn.Module):
    """MiniUnet的下采样模块
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

        # 上采样
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # 上采样
        x = self.upsample(x)

        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = x + res

        return x


class MiddleBlock(nn.Module):
    """MiniUnet的中间模块
    """
    def __init__(self, in_channels, out_channels):
        super(MiddleBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = x + res

        return x


class MiniUnet(nn.Module):
    """采用MiniUnet，对MNIST数据做生成
        两个下采样层 一个中间层 两个上采样层
    """
    def __init__(self):
        super(MiniUnet, self).__init__()

        self.conv_in = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        self.down1 = DownBlock(16, 16)
        self.down2 = DownBlock(16, 32)
        self.middle = MiddleBlock(32, 64)
        self.up1 = UpBlock(64, 32)
        self.up2 = UpBlock(32, 16)

        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    # 对时间进行正弦函数的编码
    def time_emb(self, t, dim):
        """对时间进行正弦函数的编码，单一维度

        Args:
            t (float): 时间，维度为[B]
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的时间，维度为[B, dim]
        """
        # 生成正弦编码
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2))
        sin_emb = torch.sin(t[:, None] / freqs[None])
        cos_emb = torch.cos(t[:, None] / freqs[None])

        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t):
        # 时间编码加上
        x = self.conv_in(x)
        t_emb = self.time_emb(t, 16)
        x = x + t_emb[:, :, None, None]
        # 下采样
        x1 = self.down1(x)
        x2 = self.down2(x1)
        # 中间层
        x3 = self.middle(x2)
        # 上采样
        x3 = x3 + x2
        x4 = self.up1(x3)
        x4 = x4 + x1
        x5 = self.up2(x4)

        x = self.conv_out(x5)
        return x5


if __name__ == '__main__':
    model = MiniUnet()
    x = torch.randn(2, 1, 28, 28)
    t = torch.randn(2)
    out = model(x, t)
    print(out.shape)
    # torch.Size([2, 16, 28, 28])
