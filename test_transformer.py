import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
# from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from models.transformer import Transformer
from utils.data_utils import TimeSeriesDataset
from utils.utils import set_seed , plot_predictions , calculate_model_metrics, save_results

# 定义命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Test Transformer Model")
    parser.add_argument("--input_dim", type=int, default=12, help="Input feature dimension")
    parser.add_argument("--model_dim", type=int, default=512, help="Model hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=1, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=1, help="Number of decoder layers")
    parser.add_argument("--output_window", type=int, default=96, help="Output sequence length")
    parser.add_argument("--input_window", type=int, default=96, help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--test_data_path", type=str, default="./data/original/test_data.csv", help="Path to test data")
    parser.add_argument("--train_data_path", type=str, default="./data/train_data.csv", help="Path to train data")
    parser.add_argument("--model_path", type=str, default="./save/transformer/", help="Path to trained model")
    parser.add_argument("--model_suffix", type=str, default="", help="Model suffix name")
    parser.add_argument("--gpu", type=int, default=7, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_csv_path", type=str, default="./results/transformer/", help="Path to save test results (.npy file)")
    parser.add_argument("--save_plot_path", type=str, default="./results/transformer/", help="Path to save predictions vs targets plot")
    return parser.parse_args()

# 主函数
def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(args.model_path, f"transformer_model_best_{args.model_suffix}.pth")
    

    # 加载训练数据
    train_data = pd.read_csv(args.train_data_path)
    train_dataset = TimeSeriesDataset(
        data=train_data,
        input_seq_len=args.input_window,
        output_seq_len=args.output_window,
        is_training=True  
    )

    feature_scaler = train_dataset.feature_scaler

    # 加载测试数据
    test_data = pd.read_csv(args.test_data_path)
    test_dataset = TimeSeriesDataset(
        data=test_data,
        input_seq_len=args.input_window,
        output_seq_len=args.output_window,
        feature_scaler=feature_scaler,
        is_training=False 
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = Transformer(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        output_window=args.output_window,
        device=device
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # for param in model.parameters():
    #     print (param)

    mse, mae, predictions, targets, inputs, mse_list, _ = calculate_model_metrics(model, test_loader, device)
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')
    
    # model_suffix = "/tramsformer/" + args.model_suffix
    # 可视化预测结果
    plot_predictions(
        inputs=inputs, 
        predictions=predictions,
        targets=targets,
        mse_list=mse_list,
        save_path=args.save_plot_path,
        model_suffix=args.model_suffix,
        scaler=train_dataset.feature_scaler
    )
    
    # save results
    # result_path = os.path.join(args.save_csv_path, args.model_suffix)
    result_path = args.save_csv_path
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    args.output_seq_len = args.output_window
    save_results(
        args=args,
        mse=mse,
        mae=mae,
        result_path=result_path
    )

if __name__ == "__main__":
    main()