import torch
import torch.nn as nn


# MiniUnet MNIST 28*28 4090 3G左右显存
class DownLayer(nn.Module):
    """MiniUnet的下采样层 Resnet
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
                 downsample=False):
        super(DownLayer, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        # 归一化层
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
        x += self.fc(temb)[:, :, None, None]  # [B, in_channels, H, W]
        x = self.conv1(x) # [B, out_channels, H, W]
        x = self.bn1(x) # [B, out_channels, H, W]
        x = self.act(x) # [B, out_channels, H, W]
        x = self.conv2(x) # [B, out_channels, H, W]
        x = self.bn2(x) # [B, out_channels, H, W]
        x = self.act(x) # [B, out_channels, H, W]

        if self.shortcut is not None:
            res = self.shortcut(res)

        x = x + res
        # [B, out_channels, H, W]
        if self.downsample:
            x = self.pool(x)
        # [B, out_channels, H/2, W/2]
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
                      downsample=False),
            DownLayer(base_channels * 2,
                      base_channels * 2,
                      time_emb_dim=self.time_emb_dim)
        ])
        self.maxpool1 = nn.MaxPool2d(2)

        self.down2 = nn.ModuleList([
            DownLayer(base_channels * 2,
                      base_channels * 4,
                      time_emb_dim=self.time_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 4,
                      base_channels * 4,
                      time_emb_dim=self.time_emb_dim)
        ])
        self.maxpool2 = nn.MaxPool2d(2)

        self.middle = MiddleLayer(base_channels * 4,
                                  base_channels * 4,
                                  time_emb_dim=self.time_emb_dim)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = nn.ModuleList([
            UpLayer(
                base_channels * 8,  # concat
                base_channels * 2,
                time_emb_dim=self.time_emb_dim,
                upsample=False),
            UpLayer(base_channels * 2,
                    base_channels * 2,
                    time_emb_dim=self.time_emb_dim)
        ])
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = nn.ModuleList([
            UpLayer(base_channels * 4,
                    base_channels,
                    time_emb_dim=self.time_emb_dim,
                    upsample=False),
            UpLayer(base_channels,
                    base_channels,
                    time_emb_dim=self.time_emb_dim)
        ])

        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    def time_emb(self, t, dim):
        """对时间进行正弦函数的编码，单一维度
       目标：让模型感知到输入x_t的时刻t
       实现方式：多种多样
       输入x：[B, C, H, W] x += temb 与空间无关的，也即每个空间位置（H, W）,都需要加上一个相同的时间编码向量[B, C]
       假设B=1 t=0.1
       1. 简单粗暴法
       temb = [0.1] * C -> [0.1, 0.1, 0.1, ……]
       x += temb.reshape(1, C, 1, 1)
       2. 类似绝对位置编码方式
       本代码实现方式
       3. 通过学习的方式（保证T是离散的0， 1， 2， 3，……，T）
       temb_learn = nn.Parameter(T+1, dim)
       x += temb_learn[t, :].reshape(1, C, 1, 1)
       
       
        Args:
            t (float): 时间，维度为[B]
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的时间，维度为[B, dim]  输入是[B, C, H, W]
        """
        # 生成正弦编码
        # 把t映射到[0, 1000]
        t = t * 1000# [B]
        # 10000^k k=torch.linspace……
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device) # [dim/2]
        # sin_emb = sin(t*1000/10000^(0, 1, 2, …… dim/2-1))
        sin_emb = torch.sin(t[:, None] / freqs) # [B, dim/2]
        cos_emb = torch.cos(t[:, None] / freqs) # [B, dim/2]
        # 位置编码包括正弦和余弦部分
        return torch.cat([sin_emb, cos_emb], dim=-1) # [B, dim]

    def label_emb(self, y, dim):
        """对类别标签进行编码，同样采用正弦编码

        Args:
            y (torch.Tensor): 图像标签，维度为[B] label:0-9
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的标签，维度为[B, dim]
        """
        y = y * 1000

        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t, y=None):
        """前向传播函数

        Args:
            x (torch.Tensor): 输入数据，维度为[B, C, H, W]
            t (torch.Tensor): 时间，维度为[B]
            y (torch.Tensor, optional): 数据标签（每一个标签是一个类别int型）或text文本（下一版本支持）,维度为[B]或[B, L]。 Defaults to None.
        """
        # x:(B, C, H, W)
        # 时间编码加上
        x = self.conv_in(x)# [B, base_channels, H, W]
        # 时间编码
        temb = self.time_emb(t, self.base_channels) # [B, base_channels]
        # 这里注意，我们把temb和labelemb加起来，作为一个整体的temb输入到MiniUnet中，让模型进行感知！二者编码维度一样，可以直接相加！就把label的条件信息融入进去了！
        if y is not None:
            # 判断y是label还是token
            if len(y.shape) == 1:
                # label编码，-1表示无条件生成，仅用于训练区分，推理的时候不需要
                # 把y中等于-1的部分找出来不进行任何编码，其余的进行编码
                yemb = self.label_emb(y, self.base_channels)
                # 把y等于-1的index找出来，然后把对应的y_emb设置为0
                yemb[y == -1] = 0.0
                temb += yemb
            else:  # 文字版本
                # 假设y是一个文本序列，使用简单的嵌入方式
                # 这里可以使用nn.Embedding或者其他文本编码方式
                embedding_dim = self.base_channels
                text_embedding = nn.Embedding(1000, embedding_dim).to(y.device)  # 假设词表大小为1000
                yemb = text_embedding(y.long())  # [B, L, embedding_dim]
                yemb = yemb.mean(dim=1)  # 对序列维度取平均，得到[B, embedding_dim]
                temb += yemb
        # 下采样
        # [B, base_channels, H, W]
        for layer in self.down1:
            x = layer(x, temb)
        # [B, base_channels*2, H, W]
        x1 = x
        x = self.maxpool1(x)
        # [B, base_channels*2, H/2, W/2]
        for layer in self.down2:
            x = layer(x, temb)
        # [B, base_channels*4, H/2, W/2]
        x2 = x
        x = self.maxpool2(x)
        # print(x.shape)
        # [B, base_channels*4, H/4, W/4]
        # 中间层
        x = self.middle(x, temb)
        # [B, base_channels*4, H/4, W/4]
        # 上采样
        # print(x.shape)
        x = torch.cat([self.upsample1(x), x2], dim=1)
        for layer in self.up1:
            x = layer(x, temb)
        x = torch.cat([self.upsample2(x), x1], dim=1)
        for layer in self.up2:
            x = layer(x, temb)

        x = self.conv_out(x)
        # print(x.shape)
        return x


if __name__ == '__main__':
    device = 'cuda'
    model = MiniUnet()
    model = model.to(device)
    x = torch.randn(2, 1, 28, 28).to(device)
    t = torch.randn(2).to(device)
    y = torch.tensor([1, 2]).to(device)

    out = model(x, t, y)
    # print(out.shape)

    # torch.Size([2, 16, 28, 28])
