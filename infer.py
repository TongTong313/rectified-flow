import torch
from model import MiniUnet
from rectified_flow import RectifiedFlow
import cv2
import os


def infer(
        checkpoint_path,
        base_channels=16,
        step=50,  # 采样步数（Euler方法的迭代次数） 10步效果就很好 1步效果不好
        num_imgs=5,
        y=None,
        cfg_scale=7.0,
        save_path='./results',
        device='cuda'):
    """flow matching模型推理

    Args:
        checkpoint_path (str): 模型路径
        base_channels (int, optional): MiniUnet的基础通道数，默认值为16。
        step (int, optional): 采样步数（Euler方法的迭代次数），默认值为50。
        num_imgs (int, optional): 推理一次生成图片数量，默认值为5。
        y (torch.Tensor, optional): 条件生成中的条件，可以为数据标签（每一个标签是一个类别int型）或text文本（下一版本支持）,维度为[B]或[B, L]，其中B要么与num_imgs相等，要么为1（所有图像依照同一个条件生成）。
        cfg_scale (float, optional): Classifier-free Guidance的缩放因子，默认值为7.0，y如果是None，无论这个值是几都是无条件生成。
        save_path (str, optional): 保存路径，默认值为'./results'。
        device (str, optional): 推理设备，默认值为'cuda'。
    """
    os.makedirs(save_path, exist_ok=True)
    if y is not None:
        assert len(y.shape) == 1 or len(
            y.shape) == 2, 'y must be 1D or 2D tensor'
        assert y.shape[0] == num_imgs or y.shape[
            0] == 1, 'y.shape[0] must be equal to num_imgs or 1'
        if y.shape[0] == 1:
            y = y.repeat(num_imgs, 1).reshape(num_imgs)
        y = y.to(device)
    # 生成一些图片
    # 加载模型
    model = MiniUnet(base_channels=base_channels)
    model.to(device)
    model.eval()

    # 加载RectifiedFlow
    rf = RectifiedFlow()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # with torch.no_grad():  # 无需梯度，加速，降显存
    with torch.no_grad():
        # 无条件或有条件生成图片
        for i in range(num_imgs):
            print(f'Generating {i}th image...')
            # Euler法间隔
            dt = 1.0 / step

            # 初始的x_t就是x_0，标准高斯噪声
            x_t = torch.randn(1, 1, 28, 28).to(device)

            for j in range(step):
                if j % 10 == 0:
                    print(f'Generating {i}th image, step {j}...')
                t = j * dt
                t = torch.tensor([t]).to(device)

                if y is not None:
                    y_i = y[i].unsqueeze(0)
                    # classifier-free guidance需要同时预测有条件和无条件的输出
                    v_pred_uncond = model(x=x_t, t=t)
                    v_pred_cond = model(x=x_t, t=t, y=y_i)
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond -
                                                          v_pred_uncond)
                else:
                    v_pred = model(x=x_t, t=t, y=None)

                # 使用Euler法计算下一个时间步长的值
                x_t = rf.euler(x_t, v_pred, dt)

            # 最后一步的x_t就是生成的图片
            # 先去掉batch维度
            x_t = x_t[0]
            # 归一化到0到1
            # x_t = (x_t / 2 + 0.5).clamp(0, 1)
            x_t = x_t.clamp(0, 1)
            img = x_t.detach().cpu().numpy()
            img = img[0] * 255
            img = img.astype('uint8')
            cv2.imwrite(os.path.join(save_path, f'{i}.png'), img)


if __name__ == '__main__':
    infer(checkpoint_path='./checkpoints/cfg/miniunet_19.pth',
          base_channels=64,
          step=30,
          num_imgs=10,
          y=torch.tensor([8]),
          cfg_scale=12.0,
          device='mps')
