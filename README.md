# Bike Share 项目

本项目旨在使用深度学习模型（如LSTM和Transformer）预测共享单车的使用情况。项目结构如下：

## 文件结构

- **data/**
  - **origin/**          # 原始数据存放目录
- **models/**
  - **lstm/**            # LSTM 模型相关文件
  - **transformer/**     # Transformer 模型相关文件
- **results/**
  - **lstm/**            # LSTM 模型的结果文件
  - **transformer/**     # Transformer 模型的结果文件
- **save/**
  - **lstm/**            # LSTM 模型的保存文件
  - **transformer/**     # Transformer 模型的保存文件
- **scripts/**           # 脚本文件存放目录
- **utils/**             # 工具函数或工具类存放目录

## QuickStart

### 训练和评估模型

运行以下脚本以训练和评估 LSTM 模型：

```bash
# 输出长度为96
bash scripts/train_lstm_96.sh
# 输出长度为240
bash scripts/train_lstm_240.sh
# 训练改进lstm
bash scripts/train_other_lstm.sh
```
运行以下脚本以训练和评估 Transformer 模型：
```bash
# 输出长度为96
bash scripts/train_transformer_96.bash
# 输出长度为240
bash scripts/train_transformer_240.bash
# 其他Transformer模型的改进版本
bash scripts/train_xxx_transformer_xxx.bash
...
```

## 评估指标

 - MSE (均方误差)
 - MAE (平均绝对误差)
 - 可视化预测结果

## 结果

模型的预测结果保存在results/目录下,包括:

 - 训练损失曲线
 - 预测结果可视化
 - 评估指标记录

## 依赖项
 - Python 3.7+
 - PyTorch
 - NumPy
 - Pandas
 - Matplotlib
 - tqdm

## 相关文档

更多详细信息,请参考各模块的代码注释和文档。

 - data_utils.py: 数据预处理工具
 - utils.py: 通用工具函数
 - models: 模型定义及实现
