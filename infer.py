import torch
from model import MiniUnet
from rectified_flow import RectifiedFlow
import cv2
import os


def infer(checkpoint_path,
          base_channels=16,
          step=50,
          num_imgs=5,
          save_path='./results'):
    # 生成一些图片
    # 加载模型
    model = MiniUnet(base_channels=base_channels)
    model.to('cuda')
    model.eval()

    # 加载RectifiedFlow
    rf = RectifiedFlow()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # 生成图片
    for i in range(num_imgs):
        print(f'Generating {i}th image...')
        # Euler法间隔
        dt = 1.0 / step

        # 初始的x_t就是x_0，标准高斯噪声
        x_t = torch.randn(1, 1, 28, 28).to('cuda')

        for j in range(step):
            if j % 10 == 0:
                print(f'Generating {i}th image, step {j}...')
            t = j * dt
            t = torch.tensor([t]).to('cuda')

            v_pred = model(x_t, t)

            # 使用Euler法计算下一个时间步长的值
            x_t = rf.euler(x_t, v_pred, dt)

        # 最后一步的x_t就是生成的图片
        # 先去掉batch维度
        x_t = x_t[0]
        # 归一化到0到1
        x_t = (x_t / 2 + 0.5).clamp(0, 1)
        img = x_t.detach().cpu().numpy()
        img = img[0] * 255
        img = img.astype('uint8')
        cv2.imwrite(os.path.join(save_path, f'{i}.png'), img)


if __name__ == '__main__':
    infer('./checkpoints/miniunet_60.pth',
          base_channels=64,
          step=10,
          num_imgs=50)
