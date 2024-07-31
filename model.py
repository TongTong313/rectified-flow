import torch
import torch.nn as nn


# MiniUnet MNIST 28*28
class DownLayer(nn.Module):
    """MiniUnet的下采样层
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
                 downsample=False):
        super(DownLayer, self).__init__()

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

        # 线性层，用于时间编码换通道 [B, dim] -> [B, in_channels]
        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        # 降采样
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2)

        self.in_channels = in_channels

    def forward(self, x, temb):
        # x: [B, C, H, W]
        res = x

        x += self.fc(temb)[:, :, None, None]  # [B, in_channels, 1, 1]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)

        x = x + res

        if self.downsample:
            x = self.pool(x)

        return x


class UpLayer(nn.Module):
    """MiniUnet的上采样层
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
                 upsample=False):
        super(UpLayer, self).__init__()

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

        # 线性层，用于时间编码换通道
        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, temb):
        # 上采样
        if self.upsample:
            x = self.upsample(x)
        res = x

        x += self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res

        return x


class MiddleLayer(nn.Module):
    """MiniUnet的中间层
    """

    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer, self).__init__()

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

        # 线性层，用于时间编码换通道
        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        res = x

        x += self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            x = self.shortcut(x)
        x = x + res

        return x


class MiniUnet(nn.Module):
    """采用MiniUnet，对MNIST数据做生成
        两个下采样block 一个中间block 两个上采样block
    """

    def __init__(self, base_channels=16, time_emb_dim=None):
        super(MiniUnet, self).__init__()

        if time_emb_dim is None:
            self.time_emb_dim = base_channels

        self.base_channels = base_channels

        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        # 多个Layer构成block
        self.down1 = nn.ModuleList([
            DownLayer(base_channels,
                      base_channels * 2,
                      time_emb_dim=self.time_emb_dim,
                      downsample=True),
            DownLayer(base_channels * 2,
                      base_channels * 2,
                      time_emb_dim=self.time_emb_dim)
        ])

        self.down2 = nn.ModuleList([
            DownLayer(base_channels * 2,
                      base_channels * 4,
                      time_emb_dim=self.time_emb_dim,
                      downsample=True),
            DownLayer(base_channels * 4,
                      base_channels * 4,
                      time_emb_dim=self.time_emb_dim)
        ])

        self.middle = MiddleLayer(base_channels * 4,
                                  base_channels * 4,
                                  time_emb_dim=self.time_emb_dim)

        self.up1 = nn.ModuleList([
            UpLayer(
                base_channels * 8,  # concat
                base_channels * 2,
                time_emb_dim=self.time_emb_dim,
                upsample=True),
            UpLayer(base_channels * 2,
                    base_channels * 2,
                    time_emb_dim=self.time_emb_dim)
        ])
        self.up2 = nn.ModuleList([
            UpLayer(base_channels * 4,
                    base_channels,
                    time_emb_dim=self.time_emb_dim,
                    upsample=True),
            UpLayer(base_channels,
                    base_channels,
                    time_emb_dim=self.time_emb_dim)
        ])

        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    # 对时间进行正弦函数的编码
    def time_emb(self, t, dim):
        """对时间进行正弦函数的编码，单一维度

        Args:
            t (float): 时间，维度为[B]
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的时间，维度为[B, dim]  输入是[B, C, H, W]
        """
        # 生成正弦编码
        # 把t映射到[0, 1000]
        t = t * 1000

        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t):
        # x:(B, C, H, W)
        # 时间编码加上
        x = self.conv_in(x)
        temb = self.time_emb(t, self.base_channels)

        # 下采样
        for layer in self.down1:
            x = layer(x, temb)
        x1 = x
        for layer in self.down2:
            x = layer(x, temb)
        x2 = x

        # 中间层
        x = self.middle(x2, temb)

        # 上采样
        x = torch.cat([x, x2], dim=1)
        for layer in self.up1:
            x = layer(x, temb)
        x = torch.cat([x, x1], dim=1)
        for layer in self.up2:
            x = layer(x, temb)

        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    model = MiniUnet()
    model = model.to('cuda')
    x = torch.randn(2, 1, 28, 28).to('cuda')
    t = torch.randn(2).to('cuda')

    out = model(x, t)
    print(out.shape)
    # torch.Size([2, 16, 28, 28])
