#!/bin/bash

# 设置 seed 列表
SEEDS=(5424 510 6657 74751 42)
GPU=2

# 设置输出文件路径
OUTPUT_FILE="results/transformer/train_transformer_96_5iter.txt"

# 创建输出目录（如果不存在）
mkdir -p "$(dirname "$OUTPUT_FILE")"

# 将脚本的所有输出重定向到文件
exec > >(tee "$OUTPUT_FILE") 2>&1

# 循环 5 次
for i in {1..4}
do
    # 获取当前循环的 seed 值
    SEED=${SEEDS[$i]}
    echo "Running iteration $i with seed=$SEED..."

    # 训练命令
    echo "Training..."
    python train_transformer.py \
        --model_dim=512 \
        --num_heads=8 \
        --num_encoder_layers=1 \
        --num_decoder_layers=1 \
        --output_window=96 \
        --batch_size=64 \
        --num_epochs=100 \
        --learning_rate=1e-4 \
        --train_data_path="data/train_data.csv" \
        --val_data_path="data/val_data.csv" \
        --save_path="save/transformer/" \
        --model_suffix="96_${i}" \
        --gpu=$GPU \
        --seed=$SEED

    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        echo "Training failed for iteration $i. Exiting..."
        exit 1
    fi

    # 测试命令
    echo "Testing..."
    python test_transformer.py \
        --model_dim=512 \
        --num_heads=8 \
        --num_encoder_layers=1 \
        --num_decoder_layers=1 \
        --output_window=96 \
        --batch_size=64 \
        --test_data_path="data/original/test_data.csv" \
        --train_data_path="data/train_data.csv" \
        --model_suffix="96_${i}" \
        --gpu=$GPU \
        --seed=$SEED 

    # 检查测试是否成功
    if [ $? -ne 0 ]; then
        echo "Testing failed for iteration $i. Exiting..."
        exit 1
    fi

    echo "Iteration $i completed."
done

echo "All iterations completed."