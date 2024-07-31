# 从零手搓Flow Matching（Rectified Flow）代码
作者：Tong Tong 

B站主页：[Tong发发](https://space.bilibili.com/100689001)

## 说明

* 本项目完全**从零手搓**，尽可能不参考其他任何代码，从论文原理出发逐步实现，因此算是**极简实现**的一种，并**不能保证最优性能**，各位大佬可以逐步修改完善，欢迎交流。
* 为了让大家都能上手，本代码只基于深度学习框架`Pytorch`和一些其他必要的库。数据集选择`MNIST`作为示例，该数据集Pytorch本身自带，数据集规模较小，也方便展示效果，最重要的是**即使是使用CPU都能训练**！！！
* 模型结构自己手搓了一个MiniUnet，大家可以根据自己的需求修改，也可以使用其他更复杂的模型，比如Unet、FPN等。
* 本套代码有相关讲解视频，详见B站：[从零手搓Flow Matching（Rectified Flow）代码](https://www.bilibili.com/video/BV1Q54y1U7Zw)
* 强烈建议先看一下本人B站关于flow matching和recitified flow的讲解视频，会更有感受

