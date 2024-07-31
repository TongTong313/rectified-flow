import torch
import os
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from model import MiniUnet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from rectified_flow import RectifiedFlow


def train(base_channels=16,
          epochs=10,
          batch_size=128,
          lr_adjust_epoch=50,
          batch_print_interval=100,
          checkpoint_save_interval=1):
    # 训练flow matching模型

    # 数据集加载
    # 把PIL转为tensor
    transform = Compose([ToTensor()])  # 变换成tensor + 变为[0, 1]

    dataset = MNIST(
        root='./data',
        train=True,  # 6w
        download=True,
        transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型加载
    model = MiniUnet(base_channels)
    model.to('cuda')

    # 优化器加载 Rectified Flow的论文里面有的用的就是AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # 学习率调整
    scheduler = StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    # RF加载
    rf = RectifiedFlow()

    # 记录训练时候每一轮的loss
    loss_list = []

    # 一些文件夹提前创建
    os.makedirs('./checkpoints', exist_ok=True)

    # 训练循环
    for epoch in range(epochs):
        for batch, data in enumerate(dataloader):
            x_1, _ = data  # x_1原始图像

            # 均匀采样[0, 1]的时间t randn 标准正态分布
            t = torch.rand(x_1.size(0))

            # 生成flow（实际上是一个点）
            x_t, x_0 = rf.create_flow(x_1, t)

            # 4090 大概占用显存3G
            x_t = x_t.to('cuda')
            x_0 = x_0.to('cuda')
            x_1 = x_1.to('cuda')
            t = t.to('cuda')

            optimizer.zero_grad()

            v_pred = model(x_t, t)

            loss = rf.mse_loss(v_pred, x_1, x_0)

            loss.backward()
            optimizer.step()

            if batch % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            loss_list.append(loss.item())

        scheduler.step()

        if epoch % checkpoint_save_interval == 0 or epoch == epochs - 1:
            # 保存模型
            save_dict = dict(model=model.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=epoch,
                             loss_list=loss_list)
            torch.save(save_dict, f'./checkpoints/miniunet_{epoch}.pth')


if __name__ == '__main__':
    train(
        base_channels=64,  # base_channels大一些有好处
        epochs=100,
        batch_size=16,  # batch_size小一些
        lr_adjust_epoch=50,
        checkpoint_save_interval=10)
