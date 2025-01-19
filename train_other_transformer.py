import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
# from sklearn.model_selection import train_test_split
from models.transformer import get_model  # 模型工厂函数
from utils.data_utils import TimeSeriesDataset  # 自定义数据集类
from utils.utils import set_seed, plot_loss, warmup_cosine  # 工具函数

# 定义命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    parser.add_argument("--model_type", type=str, default="transformer", 
                        choices=["transformer", "conv_transformer", "conv_periodpos_transformer", "periodcpos_transformer"],
                        help="Type of model to use (transformer, conv_transformer, conv_periodpos_transformer, periodcpos_transformer)")
    parser.add_argument("--input_dim", type=int, default=12, help="Input feature dimension")
    parser.add_argument("--model_dim", type=int, default=512, help="Model hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=1, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=1, help="Number of decoder layers")
    parser.add_argument("--output_window", type=int, default=96, help="Output sequence length")
    parser.add_argument("--input_window", type=int, default=96, help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_data_path", type=str, default="./data/train_data.csv", help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default="./data/val_data.csv", help="Path to validation data")
    parser.add_argument("--save_path", type=str, default="./save/transformer/", help="Model save path")
    parser.add_argument("--model_suffix", type=str, default=None, help="Model suffix name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

# 训练函数
def train(model, dataloader, criterion, optimizer, epoch_info, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc=f"Epoch [{epoch_info[0]+1}/{epoch_info[1]}] Training", unit="batch"):
        src, tgt = src.float().to(device), tgt.float().to(device)
        optimizer.zero_grad()
        output = model(src, tgt)  # 训练时使用目标序列
        loss = criterion(output, tgt[:, :, -1])  # 假设 tgt 的最后一列是目标值
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 验证函数
def validate(model, dataloader, criterion, epoch_info, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc=f"Epoch [{epoch_info[0]+1}/{epoch_info[1]}] Validation", unit="batch"):
            src, tgt = src.float().to(device), tgt.float().to(device)
            output = model(src)  # 验证时使用自回归模式
            loss = criterion(output, tgt[:, :, -1])
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 主函数
def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_data = pd.read_csv(args.train_data_path)
    val_data = pd.read_csv(args.val_data_path)

    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(
        data=train_data,
        input_seq_len=args.input_window,
        output_seq_len=args.output_window,
        is_training=True  
    )

    val_dataset = TimeSeriesDataset(
        data=val_data,
        input_seq_len=args.input_window,
        output_seq_len=args.output_window,
        feature_scaler=train_dataset.feature_scaler,  
        is_training=False  
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    model = get_model(
        model_type=args.model_type,
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        output_window=args.output_window,
        device=device
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练和验证
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(args.num_epochs):
        # 学习率调整
        warmup_cosine(
            optimizer=optimizer, 
            current_epoch=epoch, 
            max_epoch=args.num_epochs,
            lr_min=0.1 * args.learning_rate, 
            lr_max=args.learning_rate,
            warmup_epoch=0.1 * args.num_epochs
        )

        # 训练和验证
        train_loss = train(model, train_loader, criterion, optimizer, (epoch, args.num_epochs), device )
        val_loss = validate(model, val_loader, criterion, (epoch, args.num_epochs), device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = f"{args.save_path}{args.model_type}_model_best_{args.model_suffix}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter == 8:  # 早停
                print("Early stopping triggered.")
                break

    # 绘制损失曲线
    plt_save_path = f"{args.save_path}{args.model_type}_model_{args.model_suffix}.png"
    plot_loss(train_losses=train_losses, val_losses=val_losses, save_path=plt_save_path)

if __name__ == "__main__":
    main()