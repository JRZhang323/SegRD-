# SegRD++ ： 重新审视基于RD++的异常检测

训练环节分为两个阶段

第一阶段网络结构：
![period1](https://github.com/JRZhang323/SegRRD/assets/168359661/5c64113e-09cb-4154-81ab-3e7c77e3cfee)

第二阶段网络结构：
![Period2](https://github.com/JRZhang323/SegRRD/assets/168359661/3b44757c-37e7-4b47-b755-82e0e8846f03)



搭建环境
conda create -n SegRD++_env python=3.9

conda activate SegRD++_env

git clone https://github.com/JRZhang323/SegRRD

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
