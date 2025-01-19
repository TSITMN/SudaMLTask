# 文件结构

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

# QuickStart

## 训练 LSTM 模型（输出长度 96）

运行以下脚本以训练 LSTM 模型（输出长度为 96）：

```bash
bash scripts/train_lstm_96.sh
```
