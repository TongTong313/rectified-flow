import torch


class RectifiedFlow:
    def euler(self, x, v, dt):
        """ 使用欧拉方法计算下一个时间步长的值
            
        Args:
            x: 当前的值，维度为 [B, C, H, W]
            v: 当前的速度，维度为 [B, C, H, W]
            dt: 时间步长
        """
        x = x + v * dt

        return x

    def create_flow(self, x_1, t):
        """ 使用x_t = t * x_1 + (1 - t) * x_0公式构建x_0到x_1的流

            X_1是原始图像 X_0是噪声图像（服从高斯分布）
            
        Args:
            x_1: 原始图像，维度为 [B, C, H, W]
            t: 一个标量，表示时间，时间范围为 [0, 1]
        """

        # 需要一个x0，x0服从高斯噪声
        x_0 = torch.randn_like(x_1)

        # 获得xt的值
        x_t = t * x_1 + (1 - t) * x_0

        return x_t

    def mse_loss(self, v, x_1, x_0):
        """ 计算RectifiedFlow的损失函数
        L = MSE(x_1 - x_0 - v(t))

        Args:
            v: 速度，维度为 [B, C, H, W]
            x_1: 原始图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        """

        # 求loss函数，是一个MSE，最后维度是[B]
        loss = torch.mean((x_1 - x_0 - v)**2, dim=[1, 2, 3])

        return loss
