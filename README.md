# 从零手搓Flow Matching（Rectified Flow）

作者：Tong Tong 

B站主页：[Double童发发](https://space.bilibili.com/323109608)

本套代码有相关讲解视频，详见B站：[从零手搓Flow Matching（Rectified Flow）](https://www.bilibili.com/video/BV1Sjv4ezEDN/)，同时强烈建议先看一下本人B站关于[Flow Matching](https://www.bilibili.com/video/BV1Wv3xeNEds/)和[Recitified Flow](https://www.bilibili.com/video/BV19m421G7W8/)的算法讲解视频，会对理解代码有很大帮助。

**特别推荐看一下本人的[扩散模型之老司机开车理论视频](https://www.bilibili.com/video/BV1qW42197dv/)，对你理解扩散模型有很大帮助~**

**TODO**：
- [ ] v1.4版本计划增加文本条件输入（计划仅做简单实验，语言模型较大无法满足让大家都能上手的目标）
- [ ] v1.3版本计划增加distillation
- [x] 开放reflow（2-Rectified Flow）模型权重和数据
- [x] v1.2版本增加reflow
- [x] 开放v1.1版本相关模型权重文件（百度网盘形式）
- [x] v1.1版本计划增加MNIST条件生成 
- [x] v1.0开放模型预训练权重（百度网盘形式）


**一些bug修复说明**:
- 感谢B站粉丝大佬@EchozL提醒，MiniUnet编的草率了，现已更新，最高分辨率的特征也concat啦~

**温馨提示（跪求支持）：**
项目更新速度受大家支持程度的影响，最新[reflow视频](https://www.bilibili.com/video/BV14XDkYNEVN/)**点赞+投币**数目大于500，我立即爆肝更新下一期视频。此外，周一到周五晚上直播有概率手搓下一期视频代码内容，大家可以期待一下~
 * 目前视频点赞+投币进度（截止2024年12月7日）: 482/500

## 项目说明
* 本项目代码基于MNIST数据集实现算法的训练与推理，可实现有条件或无条件生成0-9手写字体，目前有条件生成仅支持使用类别label，也即0-9整型数字，使用文本作为条件计划下个版本支持。
* 本项目完全**从零手搓**，尽可能不参考其他任何代码，从论文原理出发逐步实现，因此算是**极简实现**的一种，并**不能保证最优性能**，各位大佬可以逐步修改完善。
* 为了让大家都能上手，本代码只基于深度学习框架Pytorch和一些其他必要的库。数据集选择MNIST作为案例，该数据集Pytorch本身自带，数据集规模较小，也方便展示效果，最重要的是**即使是使用CPU都能训练**！！！
* 模型结构自己手搓了一个MiniUnet，大家可以根据自己的需求修改，也可以使用其他更复杂的模型，比如Unet、DiT等。
* 代码中有很多注释，希望能帮助大家理解代码，如果有问题欢迎留言交流。
* 代码环境要求很低，甚至不需要GPU都可以
  * Python 3.8+
  * Pytorch 2.0+ 
  * Numpy
  * Matplotlib
  * 其他的就缺啥装啥
* 代码运行方式
  * 如果需要训练代码请务必先查看config文件夹里的配置文件，并根据实际情况修改相关参数，尤其是是否使用classifier-free guidance，是否使用GPU等，设置好了再开始训练
  * 训练：`python train.py`，训练参数配置文件为`config/train_config.yaml`
  * reflow训练：`python train_reflow.py`，训练参数配置文件为`config/train_reflow_config.yaml`
  * 推理：`python infer.py`
  * 画loss曲线：`python plot_loss_curve.py`
  * 结果图像展示（100张生成图像拼图生成）：`python draw_result_fig.py`
* 各版本权重代码和数据[点击下载](https://pan.baidu.com/s/1ZV1z9OSSXRYX5E5Ws8xvow?pwd=9hmi)，提取码9hmi，把checkpoints和data文件夹放到根目录下即可，**注意！代码或模型版本更新导致文件同步更新！请下载最新文件，更新日期2024年11月10日**

## 版本说明
### V1.2: Reflow
* V1.2版本在V1.1版本的基础上进一步支持reflow训练
* Reflow模型需要构建新的数据集，根据实验结果**所需数据量极大，算力成本较高，带来的提升确不够明显，对于MNIST这种简单数据集实用性不强**。6万张MNIST数据集需要**100万个**通过原生rectified flow模型（也即1-Rectified Flow模型）的样本对$`(Z_{0}^{1}, Z_{1}^{1})`$训练20个epoch，才有能看出来的效果
* Reflow过程模型初始权重为1-Rectified Flow模型的权重
* 模型收敛较好
  
![loss curve](/fig/loss_curve_cfg_reflow.png)
* 生成效果展示，每一行为一个类别的生成结果，从0-9，上图为2-Rectified Flow模型**2步**生成效果，下图为1-Rectified Flow模型的**2步**生成效果

![results](/fig/results_fig_cfg_reflow_2steps.png)
![results](/fig/results_fig_cfg_2steps.png)



### V1.1: Flow Matching(Rectified Flow)条件生成
* V1.1版本同时支持无条件生成和条件生成
* 模型收敛较好

![loss curve](/fig/loss_curve_cfg.png)
* 生成效果展示，每一行为一个类别的生成结果，从0-9

![results](/fig/results_fig_cfg.png)

### V1.0：Flow Matching(Rectified Flow)无条件生成
* V1.0版本仅支持无条件生成
* 模型收敛较好

![loss curve](/fig/loss_curve.png)
* 生成效果展示

![results](/fig/results_fig.png)

---
* 代码实现原理参考论文
    * Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
    * Flow Matching for Generative Modeling
    * Classifier-Free Diffusion Guidance
