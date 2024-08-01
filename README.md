# 从零手搓Flow Matching（Rectified Flow）代码
作者：Tong Tong 

B站主页：[Tong发发](https://space.bilibili.com/323109608)

## V1.0: Flow Matching(Rectified Flow)无条件生成

### 说明

* 本套代码有相关讲解视频，详见B站：[从零手搓Flow Matching（Rectified Flow）](https://www.bilibili.com/video/BV1Sjv4ezEDN/)，同时强烈建议先看一下本人B站关于[Flow Matching](https://www.bilibili.com/video/BV1Wv3xeNEds/)和[Recitified Flow](https://www.bilibili.com/video/BV19m421G7W8/)的算法讲解视频，会对理解代码有很大帮助。
* 代码基于MNIST数据集实现算法的训练与推理，最终可实现无条件生成0~9手写字体
* 本项目完全**从零手搓**，尽可能不参考其他任何代码，从论文原理出发逐步实现，因此算是**极简实现**的一种，并**不能保证最优性能**，各位大佬可以逐步修改完善，欢迎交流。
* 为了让大家都能上手，本代码只基于深度学习框架Pytorch和一些其他必要的库。数据集选择MNIST作为案例，该数据集Pytorch本身自带，数据集规模较小，也方便展示效果，最重要的是**即使是使用CPU都能训练**！！！
* 模型结构自己手搓了一个MiniUnet，大家可以根据自己的需求修改，也可以使用其他更复杂的模型，比如Unet、DiT等。
* 代码中有很多注释，希望能帮助大家理解代码，如果有问题欢迎留言交流。

