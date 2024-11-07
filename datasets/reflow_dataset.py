import torch
import cv2
from torch.utils.data import Dataset
from typing import List, Union, Tuple, Optional, Any
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import os


class ReflowDataset(Dataset):
    """ReflowDataset
    用于训练Reflow模型的数据集
    
    Args:
        img_root_path (str): 图像的根路径
        noise_root_path (str): 噪声的根路径
        transform (optional): 图像transform. Defaults to None.
    """

    def __init__(self,
                 img_root_path: str,
                 noise_root_path: str,
                 transform: Optional[Any] = None):
        # 通过根路径获得所有图片的路径
        self.img_path = []
        self.noise_path = []
        self.labels = []

        for label in os.listdir(img_root_path):
            img_path = os.path.join(img_root_path, label)
            noise_path = os.path.join(noise_root_path, label)
            for img_name in os.listdir(img_path):
                self.labels.append(int(label))
                self.img_path.append(os.path.join(img_path, img_name))
                self.noise_path.append(
                    os.path.join(noise_path, img_name.replace('.png', '.npy')))

        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        noise_path = self.noise_path[idx]
        label = self.labels[idx]

        # 读取png图片
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 读取npy文件
        noise = np.load(noise_path)

        if self.transform:
            img = self.transform(img)

        # noise已经自动变为tensor
        noise = torch.tensor(noise)
        # 删除一维
        noise = noise.squeeze(0)

        return {'img': img, 'noise': noise, 'label': label}


if __name__ == '__main__':
    transform = ToTensor()
    dataset = ReflowDataset('./data/reflow_img', './data/reflow_noise',
                            transform)
    img, noise, label = dataset[100]
    print(len(dataset))
    print(img.shape, noise.shape, label)
    print(img.max(), img.min())
