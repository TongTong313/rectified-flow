from infer import infer
import torch
import os

if __name__ == '__main__':
    # 每个数字生成10000张图像

    for i in range(10):
        save_path = f'./data/reflow_img/{i}'
        save_noise_path = f'./data/reflow_noise/{i}'
        y = [i] * 100000

        infer(checkpoint_path='./checkpoints/v1.1-cfg/miniunet_49.pth',
              base_channels=64,
              step=20,
              num_imgs=100000,
              y=torch.tensor(y),
              cfg_scale=7.0,
              save_path=save_path,
              save_noise_path=save_noise_path,
              device='cuda')
