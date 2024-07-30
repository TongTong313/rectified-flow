import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    # 画Loss曲线看收敛情况
    # 读取pth文件，获得loss_list
    checkpoint = torch.load('./checkpoints/miniunet_29.pth')
    loss_list = checkpoint['loss_list']

    # 画图
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()
