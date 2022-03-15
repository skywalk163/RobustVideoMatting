# 稳定视频抠像 (RVM) 飞桨代码复现版

![Teaser](/documentation/image/teaser.gif)



论文 [Robust High-Resolution Video Matting with Temporal Guidance](https://peterl1n.github.io/RobustVideoMatting/) 的官方 GitHub 库。RVM 专为稳定人物视频抠像设计。不同于现有神经网络将每一帧作为单独图片处理，RVM 使用循环神经网络，在处理视频流时有时间记忆。RVM 可在任意视频上做实时高清抠像。在 Nvidia GTX 1080Ti 上实现 **4K 76FPS** 和 **HD 104FPS**。此研究项目来自[字节跳动](https://www.bytedance.com/)。

<br>

## 更新
* [2022.3.12日] 新增加了mobilenetv3骨干网络支持。
* [2022.2.26日] 新加了飞桨PaddlePaddle代码实现。暂时只复现了ResNet50的推理部分。
* [2021年11月3日] 修复了 [train.py](https://github.com/PeterL1n/RobustVideoMatting/commit/48effc91576a9e0e7a8519f3da687c0d3522045f) 的 bug。
* [2021年9月16日] 代码重新以 GPL-3.0 许可发布。
* [2021年8月25日] 公开代码和模型。
* [2021年7月27日] 论文被 WACV 2022 收录。

<br>

## 展示视频
观看展示视频 ([YouTube](https://youtu.be/Jvzltozpbpk), [Bilibili](https://www.bilibili.com/video/BV1Z3411B7g7/))，了解模型能力。
<p align="center">
    <a href="https://youtu.be/Jvzltozpbpk">
        <img src="documentation/image/showreel.gif">
    </a>
</p>

视频中的所有素材都提供下载，可用于测试模型：[Google Drive](https://drive.google.com/drive/folders/1VFnWwuu-YXDKG-N6vcjK_nL7YZMFapMU?usp=sharing)

<br>


## Demo
* [网页](https://peterl1n.github.io/RobustVideoMatting/#/demo): 在浏览器里看摄像头抠像效果，展示模型内部循环记忆值。


<br>

## 下载
ResNet50 和mobilenetv3的模型 。


<table border="1">
  <tr>
    <th>框架</th>
    <th>下载</th>
  </tr>
  <tr>
    <td>飞桨rvm_resnet50.pdparams</td>
    <td>链接: https://pan.baidu.com/s/1wfWuqA04gnPiJ4EXF4FpWw 提取码: fs1e</td>
  </tr>
    <tr>
    <td>飞桨rvm_mobilenetv3.pdparams</td>
    <td>链接: https://pan.baidu.com/s/1O-Z8BnpypOz5uQn39n_GDw 提取码: q3af</td>
  </tr>
  

</table>

<br>

## 飞桨 范例

1. 安装 Python 库:
```sh
pip install -r requirements_inference.txt
```

2. 加载模型:

```python
import paddle
from model import MattingNetwork

model = MattingNetwork('resnet50') 
model.set_state_dict(paddle.load("rvm_resnet50.pdparams"))
```

3. 若只需要做视频抠像处理，我们提供简单的 API:

```python
import paddle
from model import MattingNetwork
from inference import convert_video
model = MattingNetwork('resnet50') 
model.set_state_dict(paddle.load("rvm_resnet50.pdparams"))
convert_video(
    model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
    input_source='dance.mp4',        # 视频文件，或图片序列文件夹
    output_type='video',             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
    output_composition='com.mp4',    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
    output_alpha="pha.mp4",          # [可选项] 输出透明度预测
    output_foreground="fgr.mp4",     # [可选项] 输出前景预测
    output_video_mbps=4,             # 若导出视频，提供视频码率
    downsample_ratio=None,           # 下采样比，可根据具体视频调节，或 None 选择自动
    seq_chunk=1                    # 设置多帧并行计算 12
)
```

没有算力也没关系，上这里[拍电影没有绿幕，AI给我们造！AIStudio项目地址](https://aistudio.baidu.com/aistudio/projectdetail/3513358)

读万卷书，不如行万里路！大家动手进项目亲自实践检验一下吧！


项目还在继续中....

