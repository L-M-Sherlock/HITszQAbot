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
├── .env # 环境配置
├── bot.py # 启动
└── config.py # 配置文件
```

## 环境

python 3.7 及以上

nlp_module/bert_pretrain 下的 pytorch_model.bin 请自行下载，详情见 [bert_pretrain/README.md](./nlp_module/bert_pretrain/README.md)

本项目没有上传训练好的模型，请自行训练

## 数据

数据文件：`./nlp_module/HITSZQA/data/train.txt`

数据格式：`question+'\t'+'\_label\_'+label`

将处理好的数据放入 `./nlp_module/HITSZQA/data/` 中替换 `train.txt`

## 训练

请移步至此项目：

https://github.com/L-M-Sherlock/Bert-Chinese-Text-Classification-Pytorch

## 预测

预测 label：`python .nlp_module/RequestHandler.py`

将需要分类的 question 放入 `rh_sub.get_result('分类句子')` 中运行，得到分类结果

## 部署

以 Ubuntu 18.04 为例

### go-cqhttp

首先，要安装 go-cqhttp，请执行以下命令：

```shell script
wget https://github.com/Mrs4s/go-cqhttp/releases/download/v1.0.0-beta2/go-cqhttp_1.0.0-beta2_linux_amd64.deb
dpkg -i go-cqhttp_1.0.0-beta2_linux_amd64.deb
```

默认情况下，go-cqhttp 已经安装到 `/usr/local/bin` 之下了。接下来我们要配置 go-cqhttp，请执行以下命令：

```shell script
cd /usr/local/bin
./go-cqhttp
```

初次运行 go-cqhttp 会自动生成配置文件。退出 go-cqhttp 后，请自行修改 config.yml

除了必填的账号和密码外，考虑到之后 NoneBot 需要通过 ws 与 go-cqhttp 通信，请将 config.yml 中的 ws-reverse 一项修改成：

```yaml
  - ws-reverse:
      # 是否禁用当前反向WS服务
      disabled: false  # 开启
      # 反向WS Universal 地址
      # 注意 设置了此项地址后下面两项将会被忽略
      universal: ws://127.0.0.1:8080/cqhttp/ws  # 端口号需要与NoneBot的PORT一致
```

以上就是 go-cqhttp 的具体配置。

### NoneBot

首先，要 clone 本项目的代码，地址任意。

然后请创建虚拟环境，再执行以下命令安装依赖包：

```shell script
pip install -r requirements.txt
```

注意：本项目由于是在 window 上测试的，所以 requirements.txt 中的 pytorch 是 cpu 版本，服务器部署若需要使用显卡，请自行修改。

待依赖安装完毕后，运行以下命令即可开启 NoneBot：

```shell script
python bot.py
```

另外，为了让 go-cqhttp 和 nonebot 通信，请在 bot.py 同级目录下建立 .env，并完善以下配置：

```editorconfig
HOST=127.0.0.1
PORT=8080
DEBUG=true
SUPERUSERS=["<管理员QQ号>"]
NICKNAME=["<BOT的昵称>"]
COMMAND_START=["/", ""]
```

上述内容仅为示例，请按需配置。