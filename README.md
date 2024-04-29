# SegRD++ ： 重新审视基于RD++的异常检测

训练环节分为两个阶段

第一阶段网络结构：
![period1](https://github.com/JRZhang323/SegRRD/assets/168359661/5c64113e-09cb-4154-81ab-3e7c77e3cfee)

第二阶段网络结构：
![Period2](https://github.com/JRZhang323/SegRRD/assets/168359661/3b44757c-37e7-4b47-b755-82e0e8846f03)



# 环境搭建

conda create -n SegRD++_env python=3.9

conda activate SegRD++_env

git clone https://github.com/JRZhang323/SegRRD

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt



# 代码运行

1. 自行下载MVTec AD数据集和Describable Textures Dataset
 
  MVTec AD ：wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz

  Describable Textures Dataset ： wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

2. 运行前请自行更改两个train文件中的mvtec_path, dtd_path, save_path的路径
   

3. 训练环节

运行训练第一阶段： python period1_train.py --gpu_id 0

运行训练第二阶段： python period2_train.py --gpu_id 0


# 部分代码来源

RD++: https://github.com/tientrandinh/revisiting-reverse-distillation
