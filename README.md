# HITszQAbot

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

## 简介

招生咨询信息问答系统是基于深度学习文本分类算法，面向招生信息咨询的 QQ 问答机器人。

## 项目结构

```none
├── bot # 基于 酷Q 的 Python 异步 QQ 机器人框架
│	├── faq
│	│	└── plugins
│	│		└── faq
│	│			└── __init__.py
│	├── config.py # 配置文件
│	└── faqbot.py # 启动
├── nn # 文本分类网络
│	├── data 
│	│	├── train.txt # 训练数据
│	│	└── vocab.pkl # 词典
│	├── model # 训练模型
│	│	├── checkpoints
│	│	└── summaries
│	│		├── dev
│	│		└── train
│	├── classifier_cnn.py # CNN 网络 
│	├── conf.ini
│	├── predict.py # 用训练好的模型预测分类标签
│	├── reClassify.py 
│	├── RequestHandler.py # 调用模型
│	├── test_pbulic.csv # 待预测的文本
│	├── setup.py
│	├── train.py # 训练模型
│	└── utils.py
├── environment.yml # 通过 conda 生成的项目依赖文件
└── requirements.txt # 通过 pip 生成的项目依赖文件
```

## 环境配置

### 方法一：通过 Anaconda 配置

假设已经安装了 Anaconda，并 clone 本仓库到本地，那么打开 cmd，输入以下命令：

```
$ cd <项目位置>
$ conda env create -f environment.yml
```

耐心等待，配置速度取决于网络环境。

### 方法二：通过 pip 配置

python 版本 3.6

打开 cmd，输入以下命令：

```
$ cd <项目位置>
$ pip install -r requirements.txt
```

注：建议使用虚拟环境

## 数据

数据文件：./nn/data/train.txt

数据格式：question+'\t'+'\_label\_'+label

将处理好的数据放入 ./nn/data 中替换 train.txt

## 训练

训练模型：python train.py 

训练好的模型存储在 ./nn/model 中

可修改的训练参数：

cnn_batch_size = 64	# 如果出现 out of memory 将参数改为 32，16 等

num_epochs = 100	# 训练的轮数，可适当增加 num_epochs 以提高分类精度

## 预测

预测 label：python RequestHandler.py

将需要分类的 question 放入 rh_sub.getResult(u'分类句子') 中运行，得到分类结果

## 部署

1. 配置 ./bot/config.py 文件
2. 下载安装[酷 Q](https://cqp.cc/) 并登入 QQ
3. 安装插件 [CoolQ HTTP API](https://cqhttp.cc/)
4. 配置 ..\酷Q Air\data\app\io.github.richardchien.coolqhttpapi\config\\*.json 文件
5. 重启酷 Q
6. 运行：python faqbot.py

更多关于酷 Q 机器人的开发与使用请参见：[基于 酷Q 的 Python 异步 QQ 机器人框架](https://nonebot.cqp.moe/)