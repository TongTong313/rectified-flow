import matplotlib.pyplot as plt
import cv2
import os

# 读取results文件夹的100张图片
img_folder = 'results'
img_files = [
    os.path.join(img_folder, f) for f in os.listdir(img_folder)
    if f.endswith('.png')
][:100]

fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))

for ax, img_file in zip(axes.flatten(), img_files):
    img = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
plt.show()
