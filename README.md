# HITszQAbot

## 简介

HITszQAbot 是基于深度学习文本分类算法，面向招生信息咨询的 QQ 问答机器人。

## 项目结构

```none
├── nlp_module # 文本分类网络
│	├── pytorch_pretrained 
│	├── ERNIE_pretrain
│	├── models 
│	│	└── bert.py # 模型
│	├── RequestHandler.py # 调用模型
│	├── run.py # 训练模型
│	├── train_eval.py # 训练过程
│	├── utils.py # 原数据处理
│	└── utils_new.py # 数据处理工具
├── src # nonebot 框架
│	└── plugins
│		├── faq
│		│	└── __init__.py # 群问答插件
│		└── txt_tools.py # 文本处理工具
├── .env
├── .env.dev # 开发环境
├── .env.prod # 生产环境
├── bot.py # 启动
└── config.py # 配置文件
```

## 环境要求

python 3.7 及以上

//TODO 待补充

## 数据

数据文件：./nlp_module/data/train.txt

数据格式：question+'\t'+'\_label\_'+label

将处理好的数据放入 .bot/nn/data 中替换 train.txt

## 训练

请移步至此项目：

https://github.com/L-M-Sherlock/Bert-Chinese-Text-Classification-Pytorch

## 预测

预测 label：python .nlp_module/RequestHandler.py

将需要分类的 question 放入 rh_sub.get_result('分类句子') 中运行，得到分类结果

## 部署

//TODO 待补充