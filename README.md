# 从零手搓Flow Matching（Rectified Flow）代码
作者：Tong Tong 

B站主页：[Tong发发](https://space.bilibili.com/323109608)

本套代码有相关讲解视频，详见B站：[从零手搓Flow Matching（Rectified Flow）](https://www.bilibili.com/video/BV1Sjv4ezEDN/)，同时强烈建议先看一下本人B站关于[Flow Matching](https://www.bilibili.com/video/BV1Wv3xeNEds/)和[Recitified Flow](https://www.bilibili.com/video/BV19m421G7W8/)的算法讲解视频，会对理解代码有很大帮助。

**特别推荐看一下本人的[扩散模型之老司机开车理论视频](https://www.bilibili.com/video/BV1qW42197dv/)，对你理解扩散模型有很大帮助~**

**TODO**：
- [x] 即将开放模型预训练权重（百度网盘形式），大家下载后可以直接运行推理代码，方便大家进行测试
- [ ] v1.1版本计划增加MNIST条件生成 


## V1.0-Flow Matching(Rectified Flow)无条件生成

### 说明

* 
* 代码基于MNIST数据集实现算法的训练与推理，可实现无条件生成0~9手写字体
* 本项目完全**从零手搓**，尽可能不参考其他任何代码，从论文原理出发逐步实现，因此算是**极简实现**的一种，并**不能保证最优性能**，各位大佬可以逐步修改完善，欢迎交流。
* 为了让大家都能上手，本代码只基于深度学习框架Pytorch和一些其他必要的库。数据集选择MNIST作为案例，该数据集Pytorch本身自带，数据集规模较小，也方便展示效果，最重要的是**即使是使用CPU都能训练**！！！
* 模型结构自己手搓了一个MiniUnet，大家可以根据自己的需求修改，也可以使用其他更复杂的模型，比如Unet、DiT等。
* 代码中有很多注释，希望能帮助大家理解代码，如果有问题欢迎留言交流。
* V1.0版本相关模型权重文件和MNIST数据集已上传至百度网盘，把checkpoints和data文件夹放到根目录下即可：
    * 链接：https://pan.baidu.com/s/1qngZgLqdOwOmSXOmRqe9EQ?pwd=svhd 
    * 提取码：svhd  
* 代码环境要求很低，甚至不需要GPU都可以
    * Python 3.8+
    * Pytorch 2.0+ 
    * Numpy
    * Matplotlib
    * 其他的就缺啥装啥
* 代码运行方式：
    * 训练：`python train.py`
    * 推理：`python infer.py`
    * 画loss曲线：`python plot_loss_curve.py`