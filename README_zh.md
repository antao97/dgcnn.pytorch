# DGCNN.pytorch
[[English]](README.md)

本仓库提供了一份PyTorch版本的 **Dynamic Graph CNN for Learning on Point Clouds (DGCNN)**(https://arxiv.xilesou.top/pdf/1801.07829)代码实现，代码框架来源于[WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn/tree/master/pytorch)。

需要注意的是，在DGCNN文章中网络结构图（图3）和下文对应的网络结构描述（第4.1节）并不吻合，我们使用PS修复了网络结构图中不吻合的地方，修改后的图如下：

&nbsp;
<p float="left">
    <img src="image/DGCNN.jpg"/>
</p>

&nbsp;

**小建议：** 3D点云实验结果往往比2D图像实验结果面临更大的随机性，因此我们建议您多跑几次实验，然后选择最佳的结果。

&nbsp;
## 运行需求
- Python 3.7
- PyTorch 1.2
- CUDA 10.0
- Python包: glob, h5py, sklearn

&nbsp;
## 内容目录
- [点云分类](#_3)
- [点云局部分割](#_8)
- [点云场景语义分割](#_13)

&nbsp;
## 点云分类
### 运行训练脚本：

- 1024点

``` 
python main_cls.py --exp_name=cls_1024 --num_points=1024 --k=20 
```

- 2048点

``` 
python main_cls.py --exp_name=cls_2048 --num_points=2048 --k=40 
```

### 训练结束后运行评估脚本：

- 1024点

``` 
python main_cls.py --exp_name=cls_1024_eval --num_points=1024 --k=20 --eval=True --model_path=checkpoints/cls_1024/models/model.t7
```

- 2048点

``` 
python main_cls.py --exp_name=cls_2048_eval --num_points=2048 --k=40 --eval=True --model_path=checkpoints/cls_2048/models/model.t7
```

### 使用提供的已训练模型运行评估脚本：

- 1024点

``` 
python main_cls.py --exp_name=cls_1024_eval --num_points=1024 --k=20 --eval=True --model_path=pretrained/model.cls.1024.t7
```

- 2048点

``` 
python main_cls.py --exp_name=cls_2048_eval --num_points=2048 --k=40 --eval=True --model_path=pretrained/model.cls.2048.t7
```

### 模型性能：
ModelNet40数据集

|  | 平均类别Acc | 整体Acc | 
| :---: | :---: | :---: | 
| 原文章（1024点） | 90.2 | 92.9 |
| 本仓库（1024点） | **90.9** | **93.3** |
| 原文章（2048点） | 90.7 | 93.5 |
| 本仓库（2048点） | **91.2** | **93.6** |

&nbsp;
## 点云局部分割
### 运行训练脚本：

- 使用数据集内全部类别

``` 
python main_partseg.py --exp_name=partseg 
```

- 选择数据集内特定类别，例如airplane

``` 
python main_partseg.py --exp_name=partseg_airplane --class_choice=airplane
```

### 训练结束后运行评估脚本：

- 使用数据集内全部类别

``` 
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=checkpoints/partseg/models/model.t7
```

- 选择数据集内特定类别，例如airplane

``` 
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=checkpoints/partseg_airplane/models/model.t7
```

### 使用提供的已训练模型运行评估脚本：

- 使用数据集内全部类别

``` 
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=pretrained/model.partseg.t7
```

- 选择数据集内特定类别，例如airplane

``` 
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=pretrained/model.partseg.airplane.t7
```

### 模型性能
ShapeNet part数据集

| | 平均IoU | Airplane | Bag | Cap | Car | Chair | Earphone | Guitar | Knife | Lamp | Laptop | Motor | Mug | Pistol | Rocket | Skateboard | Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 形状数量 | | 2690 | 76 | 55 | 898 | 3758 | 69 | 787 | 392 | 1547 | 451 | 202 | 184 | 283 | 66 | 152 | 5271 | 
| 原文章 | **85.2** | 84.0 | **83.4** | **86.7** | 77.8 | 90.6 | 74.7 | 91.2 | **87.5** | 82.8 | **95.7** | 66.3 | **94.9** | 81.1 | **63.5** | 74.5 | 82.6 |
| 本仓库 | **85.2** | **84.5** | 80.3 | 84.7 | **79.8** | **91.1** | **76.8** | **92.0** | 87.3 | **83.8** | **95.7** | **69.6** | 94.3 | **83.7** | 51.5 | **76.1** | **82.8** |

&nbsp;
## 点云场景语义分割

在此任务中网络结构和点云局部分割有细微不同，最后的一个MLP尺寸改为（512, 256, 13），而且在256后只使用一个dropout。

您必须从 https://goo.gl/forms/4SoGp4KtH1jfRqEj2 手动下载数据集`Stanford3dDataset_v1.2_Aligned_Version.zip`，然后将其放在`data/`目录下。

### 运行训练脚本：

此任务使用6折训练，因此需要训练6个模型，轮流选择数据集中6个区域中的1个作为这个模型的测试区域。

- 在区域1-5上训练

``` 
python main_semseg.py --exp_name=semseg_6 --test_area=6 
```

### 训练结束后运行评估脚本：

- 当模型在区域1-5训练完成后，在区域6中评估

``` 
python main_semseg.py --exp_name=semseg_eval_6 --test_area=6 --eval=True --model_root=checkpoints/semseg/models/
```

- 当6个模型训练完成后，在所有区域上评估

``` 
python main_semseg.py --exp_name=semseg_eval --test_area=all --eval=True --model_root=checkpoints/semseg/models/
```

### 使用提供的已训练模型运行评估脚本：

- 使用提供的在区域1-5上已训练模型，在区域6中评估

``` 
python main_semseg.py --exp_name=semseg_eval_6 --test_area=6 --eval=True --model_root=pretrained/semseg/
```

- 使用提供的6个已训练模型，在所有区域上评估

``` 
python main_semseg.py --exp_name=semseg_eval --test_area=all --eval=True --model_root=pretrained/semseg/
```

### 模型性能：
斯坦福大学大型3D室内空间数据集（S3DIS）

|  | 平均IoU | 整体Acc | 
| :---: | :---: | :---: | 
| 原文章 | 56.1 | 84.1 |
| 本仓库 | **59.2** | **85.0** |
